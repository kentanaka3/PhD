```
                               .-=+**######*+=-:.                             
                          .=*%@@@@@@@@@@@@@@@@@@@@#+-                         
                       :+%@@@@@@@@@@@@@@@@@@@@@@@@@@@@#=.                     
                    .+@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#-                   
                  .*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@=                 
                 +@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%-               
               .%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#              
       ...................................:-=++++**%@@@@@@@@@@@@%.            
    .....................................=+++++++++++#@@@@@@@@@@@@.           
  ......................................++++++++++++++#@@@@@@@@@@@@           
  .....................................-+++++++++++++++%%%%%%%%%%%%+          
  ....................                                 .............=++++-:   
   .................                                   .............:+++++++: 
    ................                                  ...............++++++++:
         ..::::::::.                                .................++++++++=
          :@@@@@@@@*.................................................++++++++=
          .@@@@@@@@@-................................................++++++++.
           %@@@@@@@@@+..............................................-++++++-  
           -@@@@@@@@@@@#=-::::::::::                ::::::::::::::::====-:    
            #@@@@@@@@@@@@@@@@@@@@@@=               :@@@@@@@@@@@@@@@=          
            .@@@@@@@@@@@@@@@@@@@@@@                *@@@@@@@@@@@@@@*           
             .@@@@@@@@@@@@@@@@@@@@+               .@@@@@@@@@@@@@@#            
              .%@@@@@@@@@@@@@@@@@@.               +@@@@@@@@@@@@@+             
                +@@@@@@@@@@@@@@@@#                @@@@@@@@@@@@@-              
                 .*@@@@@@@@@@@@@@-               =@@@@@@@@@@@+                
                   :*@@@@@@@@@@@%                %@@@@@@@@@+                  
                     .+%@@@@@@@@=               :@@@@@@@#=                    
                        :+#@@@@@                *@@@@*=.                      
                            :=*=                #+-.                          
```
# PhD
## Abstract
In this work, we developed an automatic machine learning (ML)-based pipeline
for earthquake pick detection and phase association, leveraging both 
high-performance and cloud computing infrastructures. The proposed method
applies publicly available models from the seismology community (Seisbench) and
addresses the challenges of processing large-scale seismic data by integrating
advanced computational techniques with ML models. Our preliminary results seem
to indicate that our method is capable of identifying P-wave and S-wave
arrivals, as well as to associating the detected events from stations
distributed across space and time. The pipeline is designed to process seismic
data from (but not limited to) the Istituto Nazionale di Oceanografia e di
Geofisica Sperimentale (OGS) in the region of North-Eastern Italy, over a fixed
period. It consists of several modules that preprocess seismic data, extract
features, evaluate model performance, and detect earthquake picks using
different Deep Learning (DL) models (e.g. CNN, RNN), while for phase
association, a Gaussian Mixture Model Associator (GaMMA) is utilized. The code
implementation is done in Python, employing GPU-based accelerators (via CUDA /
Pytorch) and multi-core processing libraries (MPI), and it is deployed on a
high-performance computing (HPC) cluster (Leonardo at CINECA) as well as a
cloud infrastructure (Ada Cloud at CINECA). This deployment enables evaluation
of the pipeline’s performance and scalability for optimal processing of large
seismic datasets. The results show that leveraging HPC infrastructure for
intensive training and cloud platforms for scalable deployment improves
efficiency, flexibility, and performance in real-time seismic event detection.
The pipeline enhances the detection of seismic events and significantly reduces
the time required for seismic data analysis compared to traditional methods and
CPU-based implementations. Due to the complexity and variability of seismic
data and the need for real-time processing, the combination of HPC and cloud
infrastructure is crucial for achieving optimal performance. The procedural
framework is adaptable to different datasets, ensuring seamless integration
with various seismic monitoring systems. Ultimately, the hybrid infrastructure
significantly reduces computation time while maintaining high detection
accuracy, making it a robust solution for earthquake monitoring systems and
seismic research.

## 
The PhD Workspace has the following file organization structure.

    ├── data
    │   ├── classified
    │   │   ├── 230601
    │   │   │   ├── 4P
    │   │   │   │   └── IT09A
    │   │   │   ├── CH
    │   │   │   │   ├── BERNI
    │   │   │   │   ├── ...
    │   │   │   │   └── SZER
    │   │   │   ├── ...
    │   │   │   └── OX
    │   │   │       └── MLN
    │   │   ├── ...
    │   │   └── 230603
    │   │       └── IV
    │   │           └── FVI
    │   ├── test
    │   │   ├── classified
    │   │   └── waveforms
    │   └── waveforms
    ├── doc
    │   ├── Doc
    │   └── References
    ├── img
    ├── inc
    │   ├── constants.py
    │   ├── downloader.py
    │   ├── initializer.py
    │   └── parser.py
    ├── src
    │   ├── __init__.py
    │   ├── analyzer.py
    │   ├── catalogger.py
    │   ├── picker.py
    │   └── Stations.py
    └── test
        ├── __init__.py
        ├── testAnalyzer.py
        ├── testCatalogger.py
        ├── testEnv.py
        ├── testInitializer.py
        ├── testParser.py
        └── testPicker.py

In the folder [src](src/) you will find the file 
[picker.py](src/picker.py). This file is able to be excuted pointing 
to any required directions. The program [picker.py](src/picker.py) is 
an *invasive* program, in the sense that creates files *(by default)* at the 
same level directory as the folder containing the *raw waveforms*. Therefore, 
please consider available memory space, especially when working with limited 
external devices.

An axample of this behaviour can be seen in the tree above. In the folder 
[data](data/), there exists the folder [waveforms](data/waveforms/) which 
contains the files to be analyzed by Picker pipeline. The program will 
create the folders 'processed', 'classified' and 'annotated' at the same level 
as [waveforms](data/waveforms/).

The project contains several manually implemented tests to provide a feedback 
of the reliability of the program, as well a framework to understand where 
a potential bug could be located during user execution of the program. In 
order to execute these tests, is as simple as typing the following command in 
the terminal:
```
% make testing
```
which will launch all the tests to establish the reliability known up to date. 
The data saved in the test are either randomly sampled data or they provide 
specific data to which we can cuantitatively measure the results obtained by 
the program.

## Picker
Picker allows the user to either customize by predefining the default 
behaviour or specify all the variables on demand or via a configuration file.
```
usage: downloader.py [-h] [-C [...]] [-F] [-G  [...]] [-K] [-M  [...]]
                     [-N [...]] [-S [...]] [-T] [-W  [...]] [-b BATCH] [-c]
                     [-d DIRECTORY] [-p PWAVE] [-s SWAVE]
                     [--client CLIENT [CLIENT ...]] [--denoiser] [--download] 
                     [--interactive] [--force] [--pyrocko] [--timing]
                     [-D YYMMDD YYMMDD | -J YYMMDD YYMMDD]
                     [--rectdomain min_lat max_lat min_lon max_lon |
                      --circdomain lat lon min_rad max_rad] [--silent | -v]

Process AdriaArray Dataset

options:
  -h, --help            show this help message and exit
  -C [ ...], --channel [ ...]
                        Specify a set of Channels to analyze. To allow
                        downloading data for any channel, set this option to
                        '*'.
  -F , --file           Supporting file path
  -G  [ ...], --groups  [ ...]
                        Analize the data based on a specified list
  -K , --key            Key to download the data from server.
  -M  [ ...], --models  [ ...]
                        Specify a set of Machine Learning based models
  -N [ ...], --network [ ...]
                        Specify a set of Networks to analyze. To allow
                        downloading data for any channel, set this option to
                        '*'.
  -S [ ...], --station [ ...]
                        Specify a set of Stations to analyze. To allow
                        downloading data for any channel, set this option to
                        '*'.
  -T, --train           Train the model
  -W  [ ...], --weights  [ ...]
                        Specify a set of pretrained weights for the selected
                        Machine Learning based model. WARNING: Weights which
                        are not available for the selected models will not be
                        considered
  -b BATCH, --batch BATCH
                        Batch size for the Machine Learning model
  -c , --config         JSON configuration file path to load the arguments.
                        WARNING: The arguments specified in the command line
                        will overwrite the arguments in the file.
  -d DIRECTORY, --directory DIRECTORY
                        Directory path to the raw files
  -p PWAVE, --pwave PWAVE
                        P wave threshold.
  -s SWAVE, --swave SWAVE
                        S wave threshold.
  --client CLIENT [CLIENT ...]
                        Client to download the data
  --denoiser            Enable Deep Denoiser model to filter the noise previous
                        to run the Machine Learning base model
  --download            Download the data
  --interactive         Interactive mode
  --force               Force running all the pipeline
  --pyrocko             Enable PyRocko calls
  --timing              Enable timing
  -D YYMMDD YYMMDD, --dates YYMMDD YYMMDD
                        Specify the beginning and ending (inclusive) Gregorian
                        date (YYMMDD) range to work with.
  -J YYMMDD YYMMDD, --julian YYMMDD YYMMDD
                        Specify the beginning and ending (inclusive) Julian
                        date (YYMMDD) range to work with.
  --rectdomain min_lat max_lat min_lon max_lon
                        Rectangular domain to download the data: [minimum
                        latitude] [maximum latitude] [minimum longitude]
                        [maximum longitude]
  --circdomain lat lon min_rad max_rad
                        Circular domain to download the data: [latitude]
                        [longitude] [minimum radius] [maximum radius]
  --silent              Silent mode
  -v, --verbose         Verbose mode
```

### How to get started?
An easy way to get started is by executing the following test command:

    % python src/picker.py -v --directory data/test/waveforms --interactive

This will run the few test examples of data we have considered worthwhile 
saving for testing purpouses. It will print all of the messages possible of 
the execution of the program and the user will be able to pause and interact 
indefenetely to analyze the output of the program. In order to continue to the 
next results the user must close the graph plot such that the program may 
continue executing.
### Next steps
The following command:

    % python src/picker.py -v -D 980101 980110 --directory path/to/waveforms -M PhaseNet EQTransformer -W instance original

will first try to search the existance of waveforms inside
```path/to/waveforms``` during the dates ```1998/01/01``` and ```1998/01/10```.
If no waveforms were found, an error will be raised stating the files were not
found.
Once the files have been found and read, the process will continue
and apply all the possible combinations  between the models
```Phasenet, EQTransformer``` and the pretrained weights
```instance, original```

### Download data

### High Performance Computing

### Testing


Good luck!

## Acknowledgement