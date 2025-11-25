import re

def get_fm_fp(file):

    misfit_pattern = r"\|\|fm\(x\)\|\|\^2 = ([\d.e+-]+)"
    penalty_pattern = r"\|\|fp\(x\)\|\|\^2 = ([\d.e+-]+)"
    
    with open(file, "r") as f:
        data = f.read()
    fm = re.findall(misfit_pattern, data)
    fp = re.findall(penalty_pattern, data)
    fm = list(map(float, fm))
    fp = list(map(float, fp))
    
    return fm, fp