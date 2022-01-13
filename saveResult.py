import os

def write_log(log, args):
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    log_path = os.path.join(args.log_path, args.foldname)
    f = open(log_path, mode='a')
    f.write(str(log))
    f.write('\n')
    f.close()