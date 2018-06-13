import pandas as pd
tempdataFrame=pd.DataFrame(columns=['Kernel Type','Training Size','Train Accuracy','Test Accuracy'])
tempdataFrame.to_csv("testResult.csv",index=False)