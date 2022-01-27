import sys, os
tmp_path = os.path.abspath(__file__)
tmp_path = os.path.dirname(tmp_path)
tmp_path = os.path.dirname(tmp_path)
tmp_path = os.path.join(tmp_path, 'prosail')
sys.path.append(tmp_path)
