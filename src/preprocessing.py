"""Функции предобработки данных."""
import re
import pandas as pd

YEAR_PATTERN = re.compile(r'\((\d{4})\)\s*$')


def extract_year_from_title(title: str) -> int | None:
    """Извлечь год выпуска из названия фильма формата 'Name (YYYY)'."""
    if not isinstance(title, str):
        return None
    match = YEAR_PATTERN.search(title.strip())
    return int(match.group(1)) if match else None


def ensure_min_train_per_user(train, val, test, min_per_user: int = 4):
    """Гарантировать >= min_per_user оценок в train для каждого пользователя из объединённого датасета."""
    all_users = pd.concat([train, val, test])['userId'].unique()
    train = train.copy()
    val = val.copy()
    test = test.copy()
    for user_id in all_users:
        train_user = train[train['userId'] == user_id]
        if len(train_user) >= min_per_user:
            continue
        need = min_per_user - len(train_user)
        val_user = val[val['userId'] == user_id].sort_values('timestamp')
        take_from_val = val_user.head(need)
        train = pd.concat([train, take_from_val], ignore_index=True)
        val = val.drop(take_from_val.index)
        need -= len(take_from_val)
        if need > 0:
            test_user = test[test['userId'] == user_id].sort_values('timestamp')
            take_from_test = test_user.head(need)
            train = pd.concat([train, take_from_test], ignore_index=True)
            test = test.drop(take_from_test.index)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)