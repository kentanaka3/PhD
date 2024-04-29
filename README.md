# PhD
The PhD Workspace has the following file organization structure.

    ├── data
    │   ├── processed
    │   │   ├── 20230601
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
    │   │   └── 20230603
    │   │       └── IV
    │   │           └── FVI
    │   ├── test
    │   │   ├── annotated
    │   │   ├── classified
    │   │   └── waveforms
    │   └── waveforms
    ├── doc
    │   ├── Doc
    │   └── References
    ├── img
    ├── src
    │   ├── __init__.py
    │   └── AdriaArray.py
    └── test
        ├── __init__.py
        └── test_AdriaArray.py

In the folder [src](src/) you will find the file 
[AdriaArray.py](src/AdriaArray.py). This file is able to be excuted pointing 
to any required directions. The program [AdriaArray.py](src/AdriaArray.py) is 
an *invasive* program, in the sense that creates files *(by default)* at the 
same level directory as the folder containing the *raw waveforms*. Therefore, 
please consider available memory space, especially when working with limited 
external devices.

An axample of this behaviour can be seen in the tree above. In the folder 
[data](data/), there exists the folder [waveforms](data/waveforms/) which 
contains the files to be analyzed by AdriaArray pipeline. The program will 
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

## AdriaArray Pipeline

### AdriaArray
AdriaArray allows the user to either customize by predefining the default 
behaviour or specify all the variables on demand.

    TODO: Consider file parsing

AdriaArray allows the following commands on demand:
- ```-h``` or ```--help```: show help message and exit
- ```-C``` or ```--channels```: The user requests a sequence of channels to be
analyzed. If file is not available, then a key must be provided in order to download the data.
- ```-D``` or ```--dates```: The user must define the initial date and the 
last date (inclusive) (*i.e.* range of dates) to be analyzed.
- ```-G``` or ```--groups```: Analyze the data based on *groups* which are 
categorical list of files which meet the criteria.
- ```-K``` or ```--key```: Key to have access to download the data from server.
- ```-M``` or ```--models```: Select a specific Machine Learning based model.
- ```-N``` or ```--networks```: The user requests a sequence of networks to be
analyzed. If file is not available, then a key must be provided in order to download the data.
- ```-S``` or ```--stations```: The user requests a sequence of stations to be
analyzed. If file is not available, then a key must be provided in order to download the data.
- ```-W```  or ```--weights```: Select a specific pretrained weights for the selected Machine Learning based model. WARNING: Weights which are not available 
for the selected models will not be considered.
- ```--directory```: The user defines the directory in which the raw data are
stored.
- ```-v``` or ```--verbose```: The user establishes the maximum level of
communication of the program towards the user. It is useful if the user wants 
to interact with the output of the program.

#### How to get started?
An easy way to get started is by executing the following command:

    % python src/AdriaArray.py -v --directory data/test/waveforms

This will run the few test examples of data we have considered worthwhile 
saving for testing purpouses. It will print all of the messages possible of 
the execution of the program and the user will be able to pause and interact 
indefenetely to analyze the output of the program. In order to continue to the 
next results the user must close the graph plot such that the program may 
continue executing.
#### Next steps
The following command:

    % python src/AdriaArray.py -v -D 19980101 19980110 --directory path/to/waveforms -M PhaseNet EQTransformer -W instance stead original

will try to search the existance of ```path/to/waveforms``` during the dates ```1998/01/01``` and ```1998/01/10``` and apply all the possible combinations between the models ```Phasenet, EQTransformer``` and the pretrained weights ```instance, stead, original```

Good luck!