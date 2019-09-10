
## 1.LightGBM 特点

* 分类
* 回归
* 排序


* 基于Histogram的决策树算法 —— 速度和内存的优化
	- Histogram算法基本思想：
    	- 先把连续的浮点特征值离散化成k个整数，同时构造一个宽度有k的直方图。在遍历数据的时候，根据离散化后的值作为索引在直方图中累积统计量，当遍历一次数据后，直方图累积了需要的统计量，然后根据直方图的离散值，遍历寻找最优的分割点。
    	- 使用直方图算法有很多优点。首先，最明显就是内存消耗的降低，直方图算法不仅不需要额外存储预排序的结果，而且可以只保存特征离散化后的值，而这个值一般用8位整型存储就足够了，内存消耗可以降低为原来的1/8。其次，在计算上的代价也大幅降低，预排序算法每遍历一个特征值就需要计算一次分裂的增益，而直方图算法只需要计算k次（k可以认为是常数），时间复杂度从O(#data*#feature)优化到O(k*#features)。
    	- Histogram算法并不是完美的。由于特征被离散化后，找到的并不是很精确的分割点，所以会对结果产生影响。但在不同的数据集上的结果表明，离散化的分割点对最终的精度影响并不是很大，甚至有时候会更好一点。原因是决策树本来就是弱模型，分割点是不是精确并不是太重要；较粗的分割点也有正则化的效果，可以有效地防止过拟合；即使单棵树的训练误差比精确分割的算法稍大，但在梯度提升（Gradient Boosting）的框架下没有太大的影响。
* 带深度限制的Leaf-wise的叶子生长策略 ——准确率的优化
* 直方图做差加速 ——  速度和内存的优化
* 直接支持类别型特征(Categorical Feature) ——准确率的优化
* Cache命中率优化 ——内存的优化
* 基于直方图的稀疏特征优化
* 多线程优化
	- 特征并行
	- 数据并行
	- 投票并行
* GPU支持
* 应用
	- 回归(Regression)
	- 二分类(Binary Classification)
	- 多分类(Multi-class Classification)
	- Lambdarank
* 模型评估
	-L1 loss
	-L2 loss
	-Log loss
	-Classification error rate
	-AUC
	-NDCG
	-Multi class log loss
	-Multi class error rate



强势之处：

* 更快的训练速度
* 更低的内存消耗
* 更好的准确率(泛化性能)
* 支持分布式，可以快速处理海量数据
* LightGBM 支持的数据格式有：CSV, TSV, LibSVM
* LightGBM 支持直接使用类别型特征，无需进行 One-Hot encodig，但是需要将类别型变量转换为 `int` 类型；


## 2.LightGBM 理论

原始算法论文：[这里](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf)



## 3.LightGBM 使用

**1.安装 LightGBM**

* CLI 版本
  - Win
  - Linux
  - OSX
  - Docker
  - Build MPI 版本
  - Build GPU 版本
* Python library
  - 安装依赖库
  - 安装 `lightgbm`


CLI 版本:

```shell

```

Python Library:

```shell
# 依赖库
$ pip install setuptools wheel numpy scipy scikit-learn -U 
$ pip install sklearn 
# lightgbm
$ pip install lightgbm
```
  
```python
import lightgbm as lgb
```

R package:

```r
install.packages("")
```

**2.数据接口【Python】**

* libsvm, tsv, csv, txt文本文件
* numpy 2维数组
* pandas 对象
* H2O DataTable’s Frame
* SciPy sparse matrix
* LightGBM 二进制文件

用法：

```python
import lightgbm as lgb

# 加载数据
# --------------------------------------------
# 加载文本文件数据或LightGBM二进制文件
# --------------------------------------------
train_csv = lgb.Dataset('train.csv')
train_tsv = lgb.Dataset('train.tsv')
train_svm = lgb.Dataset('train.svm')
train_bin = lgb.Dataset('train.bin')

# --------------------------------------------
# 加载numpy2维数组
# --------------------------------------------
data = np.random.rand(500, 10)
label = np.random.randint(2, size = 500)
train_array = lgb.Dataset(data, label = label)
# --------------------------------------------
# 加载scipy.sparse.csr_matrix数组
# --------------------------------------------
csr = scipy.sparse.csr_matirx((dat, (row, col)))
train_sparse = lgb.Dataset(csr)

# --------------------------------------------
# 保存数据为LightGBM二进制文件
# --------------------------------------------
train_data.save_binary('train.bin')


# --------------------------------------------
# 创建验证数据(validation data，在 LightGBM 中, 验证数据应该与训练数据一致（格式一致）
# --------------------------------------------
test_data1 = train_data.create_vaild('test_svm')
test_data2 = lgb.Dataset('test.svm', reference = train_data)

# --------------------------------------------
# 指定特征名称;
# 指定分类特征(构造Dataset之前应该将分类特征转换为int类型的值);
# 设置权重;
# 初始化score
# 设置group/query数据以用于ranking(排序)任务
# --------------------------------------------
data = np.random.rand(500, 10)
label = np.random.randint(2, size = 500)
train_array = lgb.Dataset(data, label = label)
w = np.random.rand(500, 1)

train_data = lgb.Dataset(data, 
                         label = label, 
                         feature_name = ['c1', 'c2', 'c3'], 
                         categorical_feature = ['c3'],
                         weight = w,
                         free_raw_data = True)
# or
train_data.set_weight(w)
 
train_data.set_init_score()

train_data.set_group()
```

**3.设置参数**

参数设置方式：

* 命令行参数
* 参数配置文件
* python参数字典

参数类型:

* 核心参数
* 学习控制参数
* IO参数
* 目标参数
* 度量参数
* 网络参数
* GPU参数
* 模型参数
* 其他参数

```python
param = {
    'num_levels': 31,
    'num_trees': 100,
    'objective': 'binary',
    'metirc': ['auc', 'binary_logloss']
}
```

**4.训练、保存、加载模型**

```python
# 训练模型
num_round = 10
bst = lgb.train(param,
                train_data,
                num_round,
                vaild_sets = [test_data])
# 保存模型
bst.save_model('model.txt')
json_model = bst.dump_model()
# 加载模型
bst = lgb.Booster(model_file = 'model.txt')
```


**5.交叉验证**

```python
num_round = 10
lgb.cv(param, train_data, num_round, nfold = 5)
```

**6.提前停止**

```python
bst = lgb.train(param,
                train_data,
                num_round,
                valid_sets = valid_sets,
                ealy_stooping_rounds = 10)
```

**7.预测**

* 用已经训练好的或加载的保存的模型对数据集进行预测
* 如果在训练过程中启用了提前停止, 可以用bst.best_iteration从最佳迭代中获得预测结果

```python
testing = np.random.rand(7, 10)
y_pred = bst.predict(testing, num_iteration = bst.best_iteration)
```




## 4.LightGBM API

### 4.1 Data Structure API

> * Dataset(data, label, reference, weight, ...)
> * Booster(params, train_set, model_file, ...)

### 4.2 Training API

> * train(params, train_set, num_boost_round, ...)
> * cv(params, train_ste, num_boost_round, ...)

### 4.3 Scikit-learn API

> * LGBMModel(boosting_type, num_leaves, ...)
> * LGBMClassifier(boosting_type, num_leaves, ...)
> * LGBMRegressor(boosting_type, num_leaves, ...)
> * LGBMRanker(boosting_type, num_leaves, ...)


```python
lightgbm.LGBMClassifier(boosting_type = "gbdt", # gbdt, dart, goss, rf
                        num_leaves = 31, 
                        max_depth = -1, 
                        learning_rate = 0.1,
                        n_estimators = 100,
                        subsample_for_bin = 200000,
                        objective = None, 
                        class_weight = None,
                        min_split_gain = 0.0,
                        min_child_weight = 0.001, 
                        min_child_samples = 20,
                        subsample = 1.0,
                        subsample_freq = 0,
                        colsample_bytree = 1.0,
                        reg_alpha = 0.0,
                        reg_lambda = 0.0,
                        random_state = None,
                        n_jobs = -1, 
                        silent = True,
                        importance_type = "split",
                        **kwargs)

lgbc.fit(X, y,
         sample, 
         weight = None, 
         init_score = None,
         eval_set = None,
         eval_names = None, 
         eval_sample_weight = None,
         eval_class_weight = None,
         eval_init_score = None,
         eval_metric = None,
         early_stopping_rounds = None,
         verbose = True,
         feature_name = "auto",
         categorical_feature = "auto",
         callbacks = None)

lgbc.predict(X, 
             raw_score = False,
             num_iteration = None,
             pred_leaf = False,
             pred_contrib = False,
             **kwargs)

lgbc.predict_proba(X, 
                   raw_score = False,
                   num_iteration = None,
                   pred_leaf = False,
                   pred_contrib = False,
                   **kwargs)
```


```python
lightgbm.LGBMRegressor(boosting_type = "gbdt",
                       num_leaves = 31,
                       max_depth = -1,
                       learning_rate = 0.1,
                       n_estimators = 100,
                       subsample_for_bin = 200000,
                       objective = None,
                       class_weight = None,
                       min_split_gain = 0.0,
                       min_child_weight = 0.001,
                       min_child_samples = 20,
                       subsample = 1.0,
                       subsample_freq = 0,
                       colsample_bytree = 1.0,
                       reg_alpha = 0.0,
                       reg_lambda = 0.0,
                       random_state = None,
                       n_jobs = -1,
                       silent = True,
                       importance_type = "split",
                       **kwargs)

lgbr.fit(X, y, sample_weight = None,
         init_score = None, 
         eval_set = None,
         eval_names = None,
         eval_sample_weight = None,
         eval_init_score = None,
         eval_metric = None,
         early_stopping_rounds = None,
         verbose = True,
         feature_name = "auto",
         categorical_feature = "auto",
         callbacks = None)

lgbr.predict(X, 
             raw_score = False, 
             num_iteration = None, 
             pred_leaf = False,
             pred_contrib = False,
             **kwargs)
```

### 4.4 Callbacks

> * early_stopping(stopping_round, ...)
> * print_evaluation(period, show_stdv)
> * record_evaluation(eval_result)
> * reset_parameter(**kwargs)

```python
early_stopping(stopping_round, ...)
print_evaluation(period, show_stdv)
record_evaluation(eval_result)
reset_parameter(**kwargs)
```

### 4.5 Plotting

> * plot_importance(booster, ax, height, xlim, ...)
> * plot_split_value_histogram(booster, feature)
> * plot_metric(booster, metric, ...)
> * plot_tree(booster, ax, tree_index, ...)
> * create_tree_digraph(booster, tree_index, ...)

```python
plot_importance(booster, ax, height, xlim, ...)
plot_split_value_histogram(booster, feature)
plot_metric(booster, ax, tree, index, ...)
plot_tree(booster, ax, tree_index, ...)
create_tree_digraph(booster, tree_index, ...)
```


