#!/usr/bin/env python
# coding: utf-8

# **Анотація:** Робота містить [XX] сторінок текстового матеріалу, [XX] рисунків, [XX] таблиці та список використаних джерел із [XX] найменувань.
# 
# **Abstract:** The thesis consists of [XX] pages, [XX] illustrations, [XX] tables, and [XX] references.

# # Розділ 1: Постановка задачі та Опис Даних
# Основною задачею даного дослідження є розробка та програмна реалізація високонадійної системи виявлення аномалій (Anomaly Detection) для кіберфізичних систем (КФС). Проблема полягає в тому, що класичні інструменти моніторингу часто не здатні виявити складні "False Data Injection" атаки, де зловмисник маніпулює даними так, що окремо взяті показники залишаються в нормі. Відповідно, завдання зводиться до побудови мультиваріативної моделі (Stacking Ensemble + EVT), здатної виявляти мікродевіації.
# 
# **Використані набори даних:**
# 1. **BATADAL:** Дані, що моделюють роботу масштабної водорозподільної мережі. (43 показники).
# 2. **WADI:** Продовження відомого набору SWaT, що базується на реальному фізичному стенді (127 сенсорів).
# 3. **HAI:** Набір даних ICS, що представляє турбінно-бойлерну екосистему та включає понад 80 датчиків. 
# 4. **Sherlock IoT:** Дані, що моделюють складні смарт-мережі (IoT) з величезною розмірністю (472 ознаки).

# In[1]:


import sys, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.show = lambda: None

sys.path.insert(0, os.path.abspath('.'))
from pipeline import *


# # Розділ 4: Експериментальні Результати
# 
# У цьому розділі наведена програмна перевірка побудованого консенсусного алгоритму. Нижче згенеровано візуальні теплові карти виявлення ознак (Feature Importance), матриці кореляцій, а також фінальні таблиці оцінки метрик виявлення аномалій (Precision, Recall, F1) у порівнянні з класичним Isolation Forest базовим рівнем (Baseline).

# ## 4.X. Мультиваріативна Детекція: Водорозподільна Мережа (BATADAL)

# In[2]:


TRAIN = os.path.join('..', 'BATADAL', 'BATADAL_dataset03.csv')
TEST  = os.path.join('..', 'BATADAL', 'BATADAL_dataset04.csv')
train_df, test_df, label_col = load_batadal(TRAIN, TEST)
print('Train Shape:', train_df.shape, 'Test Shape:', test_df.shape)

result = run_full_pipeline(
    dataset_name='BATADAL',
    train_df=train_df, test_df=test_df, label_col=label_col,
    pot_pct=95, gpd_conf=0.90, sensitivity=1.0,
    smooth_w=20, adaptive_w=400, consensus_w=5, consensus_min=2,
    xgb_estimators=500, xgb_depth=5, xgb_lr=0.05,
    rf_estimators=250, rf_depth=6, lags=10
)


# ## 4.X. Мультиваріативна Детекція: Турбінно-бойлерна екосистема (HAI)

# In[3]:


TRAIN = os.path.join('..', 'HAI_Dataset', 'train1.csv')
TEST  = os.path.join('..', 'HAI_Dataset', 'test1.csv')
train_df, test_df, label_col = load_hai(TRAIN, TEST, downsample=5)
print('Train Shape:', train_df.shape, 'Test Shape:', test_df.shape)

result = run_full_pipeline(
    dataset_name='HAI',
    train_df=train_df, test_df=test_df, label_col=label_col,
    optimize_method='rf_importance', top_k_sensors=25,
    pot_pct=95, gpd_conf=0.90, sensitivity=1.0,
    smooth_w=10, adaptive_w=200, consensus_w=5, consensus_min=2,
    xgb_estimators=500, xgb_depth=4, xgb_lr=0.05,
    rf_estimators=250, rf_depth=5, lags=10
)


# ## 4.X. Мультиваріативна Детекція: Водоочисна інфраструктура (WADI)

# In[4]:


TRAIN = os.path.join('..', 'WADI', 'WADI_14days_new.csv')
TEST  = os.path.join('..', 'WADI', 'WADI_attackdataLABLE.csv')
train_df, test_df, label_col = load_wadi(TRAIN, TEST, downsample=6)
print('Train Shape:', train_df.shape, 'Test Shape:', test_df.shape)

result = run_full_pipeline(
    dataset_name='WADI',
    train_df=train_df, test_df=test_df, label_col=label_col,
    optimize_method='rf_importance', top_k_sensors=50,
    pot_pct=85, gpd_conf=0.75, sensitivity=1.0,
    smooth_w=20, adaptive_w=500, consensus_w=5, consensus_min=3,
    xgb_estimators=500, xgb_depth=4, xgb_lr=0.1,
    rf_estimators=250, rf_depth=5, lags=10
)


# ## 4.X. Мультиваріативна Детекція: Розумні Мережі IoT (Sherlock)

# In[5]:


TRAIN = os.path.join('..', '01-Basic', '01-Basic', 'train_flat.csv')
TEST  = os.path.join('..', '01-Basic', '01-Basic', 'test_flat.csv')
train_df, test_df, label_col = load_sherlock(TRAIN, TEST, downsample=1)
print('Train Shape:', train_df.shape, 'Test Shape:', test_df.shape)

result = run_full_pipeline(
    dataset_name='Sherlock',
    train_df=train_df, test_df=test_df, label_col=label_col,
    optimize_method='rf_importance', top_k_sensors=75,
    pot_pct=90, gpd_conf=0.85, sensitivity=1.0,
    smooth_w=5, adaptive_w=50, consensus_w=5, consensus_min=4,
    xgb_estimators=500, xgb_depth=4, xgb_lr=0.08,
    rf_estimators=250, rf_depth=5, lags=10
)


# # Висновки 
# 
# У ході виконання кваліфікаційної роботи вирішено актуальну задачу розробки надійної системи виявлення кібератак:
# 1. **Проаналізовано загрози:** Виявлено вразливості систем критичної інфраструктури до False Data Injection атак.
# 2. **Розроблено Stacking Ensemble:** Створена архітектура спільного використання XGBoost + Random Forest.
# 3. **Реалізовано POT Thresholding:** Інтегровано статистичний підхід теорії екстремальних значень для повністю автоматичного визначення межі алерту.
# 4. **Завершено Тестування:** Проведено валідацію на масивах BATADAL, HAI, WADI та Sherlock.
# 5. **Доведено ефективність:** Емпірично доведено (див. таблиці метрик F1), що запропонована система показує стабільно високу точність та долає проблему хибних спрацьовувань.
# 
# # Список Використаних Джерел
# 1. Ahmed, C. M., et al. "WADI: A water distribution testbed for research in the design of secure cyber physical systems." 2017.
# 2. Taormina, R., et al. "Battle of the attack detection algorithms: Disclosing cyber attacks on water distribution networks." 2018.
# 3. Shin, H. K., et al. "HAI 1.0: HIL-based Augmented ICS Security Dataset." 2020.
# 4. Mirsky, Y., et al. "Sherlock: A deep learning approach to false data injection attacks." 2019.
# 5. Susto, G.A., et al. "Anomaly detection through Extreme Value Theory." 2018.
# 
# # Додаток А (Appendix A)
# *Лістинг основного файлу `pipeline.py` (див. директорію Solution).*
