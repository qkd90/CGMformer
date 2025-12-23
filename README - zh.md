# CGMformer

**CGMformer：一种用于解析连续血糖监测（CGM）数据中个体血糖动态特征的预训练 Transformer 模型。**

相关论文发表于：

Yurun Lu, Dan Liu, Zhongming Liang, 等.
 *A pretrained transformer model for decoding individual glucose dynamics from continuous glucose monitoring data.*
 National Science Review
 https://doi.org/10.1093/nsr/nwaf039

------

## 安装（Installation）

使用以下命令安装所需依赖：

```
pip install -r requirements.txt
```

从 Google Drive 下载模型权重文件（checkpoint）：

```
wget https://drive.google.com/file/d/1SOUkaRoMR7eOGb2EUYBJ-QmXI1Lc0af9
```

------

## 数据处理（Data processing）

由于不同来源的 CGM 数据在属性和格式上存在差异，**建议参考 `processing_811_data.ipynb` 文件对原始 CGM 数据进行处理**。
 在该处理流程中，连续血糖数据将被标注并存储在键名为 `"input_ids"` 的字段中，作为模型的输入。

在 `build_vocab.ipynb` 中，我们构建了一个词表（vocab），数值范围为 **39–301**，并包含以下特殊标记：

- `<MASK>`
- `<PAD>`
- `<CLS>`

该词表用于将血糖数值离散化并输入 Transformer 模型。

------

## 预训练（Pre-training）

### CGMformer 的预训练

若使用**无标签的 CGM 数据**对 CGMformer 进行预训练，可运行以下脚本：

```
deepspeed --num_gpus={num_gpus} run_pretrain_CGMFormer.py
```

其中：

- `num_gpus`：用于训练的 GPU 数量

------

### 在不进行微调的情况下获取样本嵌入向量

可以直接利用预训练好的 CGMformer 提取 CGM 数据的嵌入表示（embedding）：

```
python run_clustering.py \
  --checkpoint_path /path/to/checkpoint \
  --data_path /path/to/data \
  --save_path /path/to/save
```

该嵌入可用于后续的聚类分析、分型或下游任务。

------

## 诊断任务（Diagnosis）

用于糖尿病及相关诊断标签的分类任务，可运行：

```
python run_labels_classify.py \
  --checkpoint_path /path/to/checkpoint \
  --train_path /path/to/train_data \
  --test_path /path/to/test_data \
  --output_path /path/to/save
```

该脚本基于 CGMformer 的表示进行监督分类，用于疾病筛查或风险识别。

------

## CGMformer_C

### CGMformer_C 的训练

训练 CGMformer_C 模型需要**成对的 CGM 数据和临床数据**。
 所需的临床指标包括：

- `age`（年龄）
- `bmi`（体重指数）
- `fpg`（空腹血糖）
- `ins0`（空腹胰岛素）
- `HOMA-IS`
- `HOMA-B`
- `pg120`（餐后 120 分钟血糖）
- `hba1c`（糖化血红蛋白）
- `hdl`（高密度脂蛋白）

训练命令如下：

```
python SupervisedC.py
```

------

### CGMformer_C 指标计算

在完成模型训练并获得 CGMformer 生成的嵌入向量后，可运行以下脚本计算 CGMformer_C 指标：

```
python CalculateSC.py
```

------

## CGMformer_type（人群分型）

CGMformer_type 模块用于基于 CGM 数据进行人群亚型划分。
 该任务依赖于 CGMformer 输出的嵌入向量：

```
python Classifier.py
```

------

## CGMformer_Diet（饮食相关预测）

CGMformer_Diet 用于预测餐后血糖反应。
 训练该模型需要以下配对数据：

- CGMformer 生成的嵌入向量
- 膳食营养信息
- 餐前血糖值
- 餐后血糖值（用于监督训练）

训练或预测命令如下：

```
python PredictGlucose.py
```
