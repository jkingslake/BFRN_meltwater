# Grounding-line flux changes due to flow of surface meltwater across Antarctic ice shelves 

## Introduction 
Meltwater produced on the surface of the Antarctic Ice Sheet flows can flow across the surface of the ice for 10's or 100's of kilometers, but rarely reaches the ocean. Instead, most of it refreezes after draingin some distance. This redistributes mass. Ice shelves provide buttressing to inland ice. Different areas of ice sheets provide different amounts of buttressing. Therefore, if the buttressing provided by the ice shelf in the location is different than the buttressing provided by the ice shelf in the location where the meltwater refreezes, the net result of melting, drianage and refreezing will be a change in ice flux across the grounding line.

This project aims to quantify the impact of meltwater flow on ice flux across the grounding line.

For more details see Poster_abstract_Kingslake_final.docx, which is a poster abstract for the WAIS conference in 2020.

The matlab directory contains code written in 2019/2020 in matlab to compute the flow of meltwater across digital elevation models with depressions. This was used extensivley by Julian Spergel in his PhD thesis (https://academiccommons.columbia.edu/doi/10.7916/swez-dp81; https://doi.org/10.7916/swez-dp81) to examine the propensity of different ice shelf areas to transport water. 

The python directory contains code using a third-party library called fill-spill-merge to do a similar computation that the matlab code was designed to perform. fill-spill-merge is written in C++ and is well tested. Future work, at least for now, will be based around fill-spill-merge, as it is probably more reliable and faster than the matlab code. The directoty contains a number of jupyter notebooks that use python to call fill-spill-merge and analyze and plot the results. 

Barnes, R., Callaghan, K.L. and Wickert, A.D., 2020. Computing water flow through complex landscapes, Part 3: Fill-Spill-Merge: Flow routing in depression hierarchies. Earth Surface Dynamics Discussions, 2020, pp.1-22.