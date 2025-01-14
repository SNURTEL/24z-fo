import optuna

fdatabase = "sqlite:///IllustrisTNG_o3_T_all_steps_500_500_o3.db"
study_name = "wd_dr_hidden_lr_o3"

study = optuna.load_study(study_name=study_name, storage=fdatabase)

trial = study.best_trial
print("Best trial:  number {}".format(trial.number))
print("Loss:        %.5e" % trial.value)
print("Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
