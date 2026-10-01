# frozen_string_literal: true

require "open3"
require "pathname"
require "yaml"

class NavigateManifest
  TEXT_EXTENSIONS = %w[.md .py .sh .tex .bib .yml .yaml .json].freeze
  ASSET_EXTENSIONS = %w[.png .jpg .jpeg .pdf .svg].freeze
  MAKE_TARGET_PATTERN = /\A[A-Za-z0-9_.%+-]+[ \t]*:/
  BASH_FUNCTION_PATTERN = /\A([[:alnum:]_]+)\(\)[[:space:]]*\{/
  PYTHON_DEFINITION_PATTERN = /\A[[:space:]]*(?:async[[:space:]]+)?def[[:space:]]+([A-Za-z_][A-Za-z0-9_]*)/

  def initialize(root, tier, module_path, include_assets, scripts)
    @root_argument = root
    @tier = Integer(tier, 10)
    @module_path = module_path
    @include_assets = include_assets == "true"
    @scripts = scripts == "true"
  end

  def generate
    @root = File.realpath(@root_argument)
    validate_path!(@root)
    return YAML.dump(tier_one_manifest).sub(/\A---\n/, "") if @tier == 1

    raise "unsupported manifest tier: #{@tier}" unless @tier == 2
    raise "module path must not be empty" if @module_path.empty?
    raise "module path must be relative" if Pathname.new(@module_path).absolute?

    validate_path!(@module_path)
    @target = resolve_target
    YAML.dump(tier_two_manifest).sub(/\A---\n/, "")
  end

  private

  def validate_path!(path)
    raise "path is not valid UTF-8: #{path.inspect}" unless path.valid_encoding?
    raise "unsupported control character in path: #{path.inspect}" if path.match?(/\p{Cc}/)
  end

  def resolve_target
    target = File.realpath(File.join(@root, @module_path))
    validate_path!(target)
    relative = Pathname.new(target).relative_path_from(Pathname.new(@root))
    if relative.each_filename.first == ".."
      raise "module path escapes root boundary: #{@module_path.inspect}"
    end
    target
  rescue Errno::ENOENT, Errno::ENOTDIR
    raise "module path does not exist: #{@module_path.inspect}"
  end

  def tier_one_manifest
    {
      "schema" => "context-manifest/v2",
      "root" => ".",
      "tier" => 1,
      "authority" => { "executable" => "primary", "conflicts" => "human_review" },
      "modules" => [
        {
          "path" => "OGS", "role" => "aggregate", "entry" => ["OGS/README.md"],
          "validate" => discover_validate(File.join(@root, "OGS")), "ops" => "reviewed"
        },
        {
          "path" => "doc", "role" => "aggregate", "entry" => ["doc/README.md"],
          "validate" => discover_validate(File.join(@root, "doc")), "ops" => "reviewed"
        },
        {
          "path" => "LLM", "role" => "governance", "entry" => ["LLM/README.md"],
          "validate" => discover_validate(File.join(@root, "LLM")), "ops" => "reviewed"
        }
      ],
      "conflicts" => tier_one_conflicts,
      "navigation" => tier_one_navigation
    }
  end

  def tier_one_conflicts
    conflicts = []
    %w[OGS/README.md doc/README.md LLM/README.md].each do |entry|
      unless File.file?(File.join(@root, entry))
        conflicts << { "type" => "missing_entry_point", "target" => entry }
      end
    end
    conflicts
  end

  def tier_one_navigation
    markdown_entries = ["OGS/README.md", "doc/README.md", "LLM/README.md"]
    yaml_entries = ["OGS/conf/config.yaml"]
    {
      "markdown" => {
        "command" => "bash LLM/scripts/md_nav.sh outline FILE --depth 2",
        "entries" => navigation_entries(markdown_entries, "md_nav.sh", 2)
      },
      "yaml" => {
        "command" => "bash LLM/scripts/yaml_nav.sh outline FILE --depth 2",
        "entries" => navigation_entries(yaml_entries, "yaml_nav.sh", 2)
      }
    }
  end

  def navigation_entries(paths, navigator, depth)
    entries = []
    paths.each do |relative_path|
      full_path = File.join(@root, relative_path)
      next unless File.file?(full_path)

      entries << {
        "path" => relative_path,
        "outline" => capture_outline(full_path, relative_path, navigator, depth)
      }
    end
    entries
  end

  def capture_outline(full_path, relative_path, navigator, depth)
    script = File.join(__dir__, navigator)
    stdout, stderr, status = Open3.capture3(
      "bash", script, "outline", full_path, "--depth", depth.to_s
    )
    $stderr.write(stderr) unless stderr.empty?
    unless status.success?
      raise "#{navigator} failed for #{relative_path.inspect} (exit #{status.exitstatus})"
    end

    outline = stdout.sub(/\n+\z/, "")
    outline.empty? ? "(no recognized structure)" : outline
  end

  def discover_validate(directory)
    makefile = File.join(directory, "Makefile")
    return ["handler-validate"] unless File.file?(makefile)

    targets = []
    File.foreach(makefile) do |line|
      next unless MAKE_TARGET_PATTERN.match?(line)

      target = line.split(":", 2).first.strip
      targets << target if target.match?(/test|check|lint|validate|dry-run/)
    end
    targets = targets.uniq.sort
    targets.empty? ? ["handler-validate"] : targets
  end

  def tier_two_manifest
    files = scan_files(@target).sort
    text_files = []
    asset_files = []
    files.each do |file|
      relative = displayed_module_file(file)
      validate_path!(relative)
      extension = File.extname(file)
      text_files << relative if TEXT_EXTENSIONS.include?(extension) || File.basename(file) == "Makefile"
      asset_files << relative if @include_assets && ASSET_EXTENSIONS.include?(extension)
    end

    manifest = {
      "schema" => "context-manifest/v2",
      "root" => ".",
      "tier" => 2,
      "module" => @module_path,
      "authority" => { "executable" => "primary", "conflicts" => "human_review" },
      "inventory" => { "text" => text_files, "assets" => asset_files },
      "validation" => { "targets" => discover_validate(@target) },
      "conflicts" => tier_two_conflicts
    }
    manifest["symbols"] = extract_symbols(files) if @scripts
    manifest
  end

  def scan_files(path)
    stat = File.lstat(path)
    return visible_path?(path) && stat.file? ? [path] : [] unless stat.directory?
    return [] unless visible_path?(path)

    Dir.children(path).sort.flat_map do |name|
      child = File.join(path, name)
      child_stat = File.lstat(child)
      next [] if name.start_with?(".")
      next [] unless child_stat.directory? || child_stat.file?

      child_stat.directory? ? scan_files(child) : [child]
    end
  end

  def visible_path?(path)
    path.split(File::SEPARATOR).none? { |part| part.start_with?(".") && !part.empty? }
  end

  def displayed_module_file(file)
    suffix = Pathname.new(file).relative_path_from(Pathname.new(@target)).to_s
    suffix == "." ? @module_path : File.join(@module_path, suffix)
  end

  def tier_two_conflicts
    File.exist?(@target) ? [] : [{ "type" => "missing_target", "path" => @module_path }]
  end

  def extract_symbols(files)
    symbols = {
      "make_targets" => [],
      "bash_functions" => [],
      "python_definitions" => []
    }

    files.each do |file|
      relative = displayed_module_file(file)
      validate_path!(relative)
      case File.basename(file)
      when "Makefile"
        File.foreach(file).with_index(1) do |line, number|
          next unless MAKE_TARGET_PATTERN.match?(line)

          target = line.split(":", 2).first.strip
          symbols["make_targets"] << { "file" => relative, "line" => number, "target" => target }
        end
      else
        extension = File.extname(file)
        if extension == ".sh"
          File.foreach(file).with_index(1) do |line, number|
            match = BASH_FUNCTION_PATTERN.match(line)
            symbols["bash_functions"] << { "file" => relative, "line" => number, "function" => match[1] } if match
          end
        elsif extension == ".py"
          File.foreach(file).with_index(1) do |line, number|
            match = PYTHON_DEFINITION_PATTERN.match(line)
            symbols["python_definitions"] << { "file" => relative, "line" => number, "symbol" => match[1] } if match
          end
        end
      end
    end
    symbols
  end
end

begin
  abort "usage: navigate_manifest.rb ROOT TIER MODULE INCLUDE_ASSETS SCRIPTS" unless ARGV.length == 5

  manifest = NavigateManifest.new(*ARGV).generate
  STDOUT.write(manifest)
rescue StandardError => error
  warn "navigate: #{error.message}"
  exit 1
end