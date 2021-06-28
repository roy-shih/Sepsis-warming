# Spesis-warming
## 2021.06.27
Model: CNN
![image](https://user-images.githubusercontent.com/79713835/123583723-b1dd1880-d812-11eb-97c9-e839b223925e.png)
### Q1: How to classify between Label2 and label0
 because:
 Label 0 is recognized as 2 have 277
 Label 2 is recognized as 0 have 595
 guess:
 CNN is to compare the similarity, so if the number is stable, whether it is too high or low or normal, it is a rectangle for the image.
 When there is no sepsis, the vital signs are normal and stable;
 When there is sepsis, the vital signs index is too high or too low and it is also stable.
 Solution:
 Increase the sampling of 2;
 Solve the model's perception of absolute values 
 (maybe Join rgb e.g. the normal value, the greener the higher, the redder the lower, the bluer the lower.)
 thereby
 |         |0       | 1       |2        |
 | 0       |4978⬆  | 58      |277⬇   |
 | 1       |166      | 48       |3        |
 | 2       |595⬇   | 4       |389⬆        |



 Q2: Whether to join
 because
 0 is recognized as 1
 1 is recognized as 0
 Guess
 The modes of 1 and 0 are very similar
 Solution
 Adjust the window size of check
 Because the number of 1 recognized as 0 is greater than the number of recognized 1 as 0
 Reduce the window size of check from 5 to 3
 Turn some 1s into 0s
 thereby
  |         |0       | 1       |2        |
 | 0       |4978  | 58      |277   |
 | 1       |166⬇      | 48⬆       |3        |
 | 2       |595   | 4       |389      |

 Because this method is mainly to reduce the recognition of 1 as 0
 If 0 is recognized as 1, the index cannot be reduced
 Maybe we can add a closely watched indicator in actual application
 It is that he may develop sepsis and needs special attention so that the hospital or doctor can prepare in advance in the allocation of resources.
 As long as the identification is 1, pay close attention to and pre-allocate resources
 If it returns to 0, it’s good. If you go to 2 to develop a hospital, you can immediately have the resources to deal with it.

 Q3:
 1 is rarely recognized as 2 looks good
