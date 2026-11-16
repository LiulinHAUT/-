# - 1. 导入核心库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve, auc)
import warnings
warnings.filterwarnings('ignore')
# 2. 全局设置（解决中文乱码）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
# 3. 加载数据集
df = pd.read_csv('C:/Users/28719/Desktop/student_depression_dataset.csv')
print("数据集基本信息：")
print(f"形状：{df.shape}")
print(f"\n列名：{list(df.columns)}")
print(f"\n前5行数据：")
print(df.head())
print(f"\n数据类型：")
print(df.dtypes)
print(f"\n缺失值统计：")
print(df.isnull().sum())
# 4. 数据清洗和预处理
# 4.1 处理缺失值
def handle_missing_data(df):
    """处理缺失值"""
    # 首先将所有可能的数值列转换为数值类型
    numeric_cols_candidates = ['Academic Pressure', 'Work Pressure', 'Financial Stress', 
                               'Study Satisfaction', 'Job Satisfaction', 'Age']
    for col in numeric_cols_candidates:
        if col in df.columns:
            # 尝试转换为数值类型，无法转换的设置为NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
            print(f"已将列 '{col}' 转换为数值类型")
    # 数值型列用中位数填充
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"列 '{col}' 的缺失值已用中位数 {median_val:.2f} 填充")
    # 分类型列用众数填充
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
            df[col] = df[col].fillna(mode_val)
            print(f"列 '{col}' 的缺失值已用众数 '{mode_val}' 填充")
    
    return df
df = handle_missing_data(df)
# 4.2 标签处理
print("\n" + "="*50)
print("标签列分析：")
print("Have you ever had suicidal thoughts ? 的值分布：")
print(df['Have you ever had suicidal thoughts ?'].value_counts())
print("\nFamily History of Mental Illness 的值分布：")
print(df['Family History of Mental Illness'].value_counts())
print("\nDepression 的值分布：")
print(df['Depression'].value_counts())
# 根据数据集，我们选择 'Depression' 作为主标签
# 如果有自杀念头或家族病史，可能增加抑郁风险，我们可以创建复合特征
label_col = "Depression"
df['depression_label'] = df[label_col].map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0, 'Depressed': 1, 'Not Depressed': 0})
# 如果映射后仍有空值，检查实际值
if df['depression_label'].isnull().any():
    print(f"\nDepression 列的唯一值：{df[label_col].unique()}")
    # 手动处理不同的标签格式
    for val in df[label_col].unique():
        if pd.isna(val):
            continue
        val_lower = str(val).lower()
        if 'yes' in val_lower or 'depressed' in val_lower or '1' in val_lower:
            df.loc[df[label_col] == val, 'depression_label'] = 1
        elif 'no' in val_lower or 'not' in val_lower or '0' in val_lower:
            df.loc[df[label_col] == val, 'depression_label'] = 0
# 删除无法识别的标签行
df = df.dropna(subset=['depression_label'])
df['depression_label'] = df['depression_label'].astype(int)
print(f"\n抑郁症标签分布：{dict(df['depression_label'].value_counts())}")
print(f"抑郁症比例：{df['depression_label'].mean():.2%}")
# 4.3 创建复合风险特征
# 如果有自杀念头，抑郁风险增加
df['suicidal_thoughts'] = df['Have you ever had suicidal thoughts ?'].apply(
    lambda x: 1 if isinstance(x, str) and ('yes' in x.lower() or '1' in x) else 0
)
# 如果有家族病史，抑郁风险增加
df['family_history'] = df['Family History of Mental Illness'].apply(
    lambda x: 1 if isinstance(x, str) and ('yes' in x.lower() or '1' in x) else 0
)
# 确保所有求和列都是数值类型
sum_cols = ['Academic Pressure', 'Work Pressure', 'Financial Stress']
for col in sum_cols:
    if col in df.columns:
        # 再次确认类型转换
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # 填充可能的NaN值
        df[col] = df[col].fillna(df[col].median())
# 创建压力总分
df['total_stress'] = df[['Academic Pressure', 'Work Pressure', 'Financial Stress']].sum(axis=1)
# 创建满意度总分
df['total_satisfaction'] = df[['Study Satisfaction', 'Job Satisfaction']].sum(axis=1)
print("\n数据预处理完成！")
print(f"数据集形状：{df.shape}")
print(f"压力总分统计：")
print(df['total_stress'].describe())
