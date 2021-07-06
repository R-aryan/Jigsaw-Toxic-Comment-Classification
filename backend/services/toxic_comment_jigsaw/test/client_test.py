from backend.common.logging.console_loger import ConsoleLogger
from toxic_comment_jigsaw.application.ai.inference.prediction import Prediction
from toxic_comment_jigsaw.application.ai.training.src.preprocess import Preprocess
from toxic_comment_jigsaw.application.ai.settings import Settings
import pandas as pd

p1 = Prediction(preprocess=Preprocess(), logger=ConsoleLogger())
data = pd.read_csv(Settings.TEST_DATA)
data = data.comment_text.values
index = 1500

print("Sample Input, ", str(data[index]))
output = p1.run_inference(data[index])
print(output)
