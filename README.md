# SFTNet
SFTNet is a microexpression-based method for depression detection.
#Background and objectives:# Depression is a typical mental illness, and early screening can effectively prevent exacerbation of the condition.
Many studies have found that the expressions of depressed patients are different from those of other subjects, and microexpressions have been used in the clinical detection of mental illness.
However, there are few methods for the automatic detection of depression based on microexpressions.
\\
#Methods:# A new dataset of 156 participants(76 in the case group and 80 in the control group) was created.
All data were collected in the context of a new emotional stimulation experiment and doctor-patient conversation.
We first analyzed the Average Number of Occurrences (ANO) and Average Duration (AD) of facial expressions in the case group and the control group.
Then, we proposed a two-stream model SFTNet for identifying depression based on microexpressions, which consists of a single-temporal network(STNet) and a full-temporal network(FTNet).
STNet is used to extract features from facial images at a single time node, FTNet is used to extract features from all-time nodes, and the decision network combines the two features to identify depression through decision fusion. 
\\
#Results:# We found that the AD of all subjects was less than 20 frames (2/3 seconds) and that the facial expressions of the control group were richer.
SFTNet achieved excellent results on the emotional stimulus experimental dataset, with Accuracy, Precision and Recall of 0.873, 0.888 and 0.846, respectively.
We also conducted experiments on the doctor-patient conversation dataset, and the Accuracy, Precision and Recall were 0.829, 0.817 and 0.837, respectively.
SFTNet can also be applied to microexpression detection task with more accuracy than SOTA models.
\\
#Conclusions:# In the emotional stimulation experiment, the subjects in the case group are more likely to show negative emotions. Compared to SOTA models, our depression detection method is more accurate and can assist doctors in the diagnosis of depression.
