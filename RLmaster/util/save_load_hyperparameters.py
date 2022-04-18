import os
import argparse
import pickle

def save_hyperparameters(args : argparse.Namespace):
    log_path = os.path.join(args.logdir, args.task, args.log_name)
    # pickle-save it so that we don't need to worry about types etc
    # this we're saving that as argparse.Namespace to avoid all fiddling
    log_path_pickle = os.path.join(log_path, "hyperparameters_pickle.pkl")
    pickling_file = open(log_path_pickle, 'wb')
    pickle.dump(args, pickling_file)
    pickling_file.close()
    # we also save the parameter-values as a csv to make them plain-text and thus readable
    parameters_as_dict = vars(args)
    log_path_csv = os.path.join(log_path, "hyperparameters_pickle.csv")
    csv_file = open(log_path_csv, "w")
    for parameter in parameters_as_dict:
        csv_file.write(parameter + "," + str(parameters_as_dict[parameter]) + "\n")
    csv_file.close()


def load_hyperparameters(log_path : str):
    log_path_pickle = os.path.join(log_path, "hyperparameters_pickle.pkl")
    pickling_file = open(log_path_pickle, 'rb')
    hyperparameters = pickle.load(pickling_file)
    pickling_file.close()
    return hyperparameters




    


