"""Run the main notebook and supplementary analyses from any working directory."""
from pathlib import Path
import argparse
import json
import os


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-only', action='store_true', help='Only main CSVs and figures')
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    os.chdir(root)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    namespace={'__name__':'__main__'}
    nb=json.loads((root/'Main_Analysis.ipynb').read_text())
    for index,cell in enumerate(nb['cells']):
        if cell['cell_type']=='code':
            print(f'Notebook cell {index}',flush=True)
            exec(compile(''.join(cell['source']),f'Main_Analysis.ipynb:cell{index}','exec'),namespace)
            plt.close('all')
    if not args.main_only:
        from scripts.supplementary import main as supplementary_main
        supplementary_main()
    print('Reproduction complete. Outputs: Results/ and Plots/.',flush=True)


if __name__=='__main__':
    main()
