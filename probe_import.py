import sys, importlib.util
sys.path.insert(0, '.')
probe = sys.argv[1]
spec = importlib.util.spec_from_file_location('probe', probe)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
print('import OK:', probe)
