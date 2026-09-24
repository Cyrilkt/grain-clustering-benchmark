def check_args(args, data_dim):
    args.NIW_prior_nu = args.NIW_prior_nu or data_dim + 2
    if args.NIW_prior_nu < data_dim + 1:
        raise Exception(f'The chosen NIW nu hyperparameter need to be at least D+1 (D is the data dim). Set --NIW_prior_nu to at least {data_dim + 1}')
