import breeze.linalg._
import java.io.File
import java.nio.file.{Paths, Files}
import java.nio.charset.StandardCharsets


object Main extends App{
  var TrainDataSet = "data/train_data.csv"
  var TestDataSet = "data/test_data.csv"


  var Train = csvread(file=new File(TrainDataSet), separator=',',skipLines=1)
  var Test = csvread(file = new File(TestDataSet), separator = ',', skipLines = 1)

  var YTrain = Train(::,IndexedSeq(4)).toDenseMatrix
  var XTrain = Train(::,IndexedSeq(0, 1, 2, 3)).toDenseMatrix

  var YTest = Test(::, IndexedSeq(4)).toDenseMatrix
  var XTest = Test(::, IndexedSeq(0, 1, 2, 3)).toDenseMatrix

  var w = inv(XTrain.t * XTrain) * XTrain.t * YTrain
  csvwrite(file = new File("predictions/y_train_predicted.csv"), XTrain * w, separator = ',')
  csvwrite(file = new File("predictions/y_test_predicted.csv"), XTest * w, separator = ',')

//  validation
  var TrainError = YTrain - XTrain * w
  var TrainError_norm = TrainError.t * TrainError
  var TrainMSE = TrainError_norm / TrainError.rows.asInstanceOf[Double]

  var TestError = YTest - XTest * w
  var TestError_norm = TestError.t * TestError
  var TestMSE = TestError_norm / TestError.rows.asInstanceOf[Double]

  var content = "Train MSE error: " + TrainMSE.toString() + "\n"
  content += "Test MSE error: " + TestMSE.toString() + "\n"

  Files.write(Paths.get("evaluation/model_evaluation.txt"), content.getBytes(StandardCharsets.UTF_8))
}



