# Rule: Mandatory Pre-Modification Code Presentation and User Approval

## Strict Invariant
The agent MUST NEVER execute any code edit, file creation, deletion, or modification tool (`replace_file_content`, `write_to_file`, or file-altering shell commands) without FIRST presenting to the user in chat:
1. A clear written description of the proposed modification, including rationale, exact lines affected, and potential side-effects.
2. The exact implementation code (complete diff snippet or replacement chunk).

## Execution Sequence
1. **Analyze**: Inspect the code and determine necessary changes.
2. **Present**: Output the written description AND the exact code diff to the user.
3. **Wait for Explicit Approval**: Stop calling tools and ask for confirmation.
4. **Apply**: ONLY after the user explicitly reviews the presented code and confirms, execute the modification tool.

Sequential, high-level, or directional instructions (e.g., "proceed to X", "next", "fix Y") DO NOT constitute approval to edit. The presentation-and-approval step is non-bypassable under all circumstances.
