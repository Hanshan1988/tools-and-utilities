import lightgbm as lgb


def regular_para(para):
    n_tree = para['n_estimators']
    new_dict = {}
    for key, items in para.items():
        if key != 'n_estimators':
            new_dict[key] = items
    new_dict['verbosity'] = -1
    new_dict['min_gain_to_split'] = 0
    return n_tree, new_dict


def run_model(train_set, valid_set, para, early_stop):
    evals_result = {}
    n_tree, para = regular_para(para)
    lgbm_model = lgb.train(para, train_set,
                           verbose_eval=50,
                           num_boost_round=n_tree,
                           valid_sets=[train_set, valid_set],
                           early_stopping_rounds=early_stop,
                           evals_result=evals_result)
    return lgbm_model, evals_result

