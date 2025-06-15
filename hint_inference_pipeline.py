import pandas as pd
import os
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

DATA_ZIP = 'Hint_Inference_Project_Data.zip'

# ensure data extracted
if not os.path.exists('assistments09.csv'):
    with zipfile.ZipFile(DATA_ZIP) as z:
        z.extractall()

# --- Load ASSISTments ---
assist = pd.read_csv('assistments09.csv', encoding='latin1')
assist['response_time'] = assist['ms_first_response'] / 1000.0
assist['attempts'] = assist['attempt_count']
assist['correct'] = assist['correct']
assist['label'] = (assist['hint_count'] > 0).astype(int)
assist = assist[['response_time', 'attempts', 'correct', 'label']].dropna()

# --- Load KDD Cup 2010 ---
kdd = pd.read_csv('kdd_cup_2010_train.tsv', sep='\t', encoding='latin1')
kdd['response_time'] = kdd['Step Duration (sec)']
kdd['attempts'] = kdd['Incorrects'].fillna(0) + 1
kdd['correct'] = kdd['Correct First Attempt']
kdd['label'] = (kdd['Hints'] > 0).astype(int)
kdd = kdd[['response_time', 'attempts', 'correct', 'label']].dropna()

# --- Combine and train ---
train_df = pd.concat([assist, kdd], ignore_index=True)
X = train_df[['response_time', 'attempts', 'correct']]
y = train_df['label']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBClassifier(
    max_depth=5,
    learning_rate=0.1,
    n_estimators=100,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    eval_metric='logloss'
)
model.fit(X_train, y_train)

print('Validation report:')
val_preds = model.predict_proba(X_val)[:, 1]
print(classification_report(y_val, val_preds > 0.5))

# --- Apply to EdNet ---
ednet = pd.read_csv('ednet_kt1_sample.csv')
questions = pd.read_csv('ednet_questions.csv')[['question_id', 'correct_answer']]
ednet = ednet.merge(questions, on='question_id', how='left')
ednet['correct'] = (ednet['user_answer'] == ednet['correct_answer']).astype(int)
ednet['response_time'] = ednet['elapsed_time'] / 1000.0
ednet['attempts'] = 1
X_ednet = ednet[['response_time', 'attempts', 'correct']]
ednet['inferred_hint_probability'] = model.predict_proba(X_ednet)[:, 1]
ednet['inferred_hint_label'] = (ednet['inferred_hint_probability'] >= 0.5).astype(int)

ednet[['solving_id', 'question_id', 'inferred_hint_probability', 'inferred_hint_label']].to_csv('ednet_with_hints.csv', index=False)
print('Saved ednet_with_hints.csv')
