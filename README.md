# Holmes Text Generator

Generates text in the style of Arthur Conan Doyle's Sherlock Canon using deep learning.

## Examples
```
 "what is the name of the case which had been surely to think that i have ...
 where does the window of your barrow. it is not a few minutes that it was a station and let me was  ...
 horace harker and the subject of the station which had been surely to think ...
 the public not unnaturally and in the hall man. i have the dark front of the morning. i should have ...
 armstrong at the door. "suppose i want to shut hat avoid to chantimage." "and ne! could we have no d ...
```


## Sampling
Run `bin/sample.py` to generate random samples
There are two approaches to sampling :
- Greedy, Large values of temperature make samples more interesting but produces more spelling mistakes
- Beam Search, Values of k=3-4 produce interesting but sometimes grammatically incorrect samples; Values of k=8-10 produce repeatitive but semantically correct samples

