# Python code calling fill-spill-merge

These notebooks use python to call a C++ library called fill-spill-merge and analyze and plot the results.

More details on the fill-spill-merge library can be found here 

Barnes, R., Callaghan, K.L. and Wickert, A.D., 2020. Computing water flow through complex landscapes, Part 3: Fill-Spill-Merge: Flow routing in depression hierarchies. Earth Surface Dynamics Discussions, 2020, pp.1-22.

and

https://github.com/r-barnes/Barnes2020-FillSpillMerge

Thank you to the developers of fill-spill-merge for making this code available. 

## Installation
I followed the instructions in the fill-spill-merge repository, adjusting slightly for the location of compilers on my system: 

```
brew install gdal libomp cmake llvm
git clone --recurse-submodules -j8 https://github.com/r-barnes/Barnes2020-FillSpillMerge
cd Barnes2020-FillSpillMerge
mkdir build
cd build
cmake -D CMAKE_C_COMPILER="/opt/homebrew/Cellar/llvm/17.0.6/bin/clang" -D CMAKE_CXX_COMPILER="/opt/homebrew/Cellar/llvm/17.0.6/bin/clang++" -DUSE_GDAL=ON  -DCMAKE_BUILD_TYPE=Release ..
make -j 8
```

## Running fill-spill-merge
Run the main fill-spill-merge algorithm as follows:

```
./fsm.exe kerry_test3.dem test 0.0 
```

(I found kerry_test3.dem [here](https://github.com/r-barnes/Barnes2019-DepressionHierarchy/blob/master/test_cases/kerry_test3.dem))

