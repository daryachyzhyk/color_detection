import os
import re
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from color_combination_model import color_config

folder_path = '/Users/ivan/Downloads/color-test'
# folder_path = '/Users/ivan/Downloads/antilooks'

if __name__ == "__main__":
    regex = '([A-Z]\d+)(C\d+)([A-Z]\d+)(C\d+)'
    data_test_df = pd.DataFrame()
    for folder in os.listdir(folder_path):
        if os.path.isdir(os.path.join(folder_path, folder)):
            df_samples = pd.DataFrame()
            df_feedback = pd.DataFrame()
            for file in os.listdir(os.path.join(folder_path, folder)):
                if file == 'data_test.csv':
                    data_test_df = pd.concat([data_test_df, pd.read_csv(os.path.join(folder_path, folder, file))])
                # if file.endswith('eedback.csv'):
                #     df_feedback = pd.read_csv(os.path.join(folder_path, folder, file))
                #
                # if file.endswith('.html'):  # == 'looks_for_clients_FR.html'
                #     list_pairs = []
                #     country_code = file.split("_")[-1].split(".")[0]
                #     with open(os.path.join(folder_path, folder, file), 'r', errors="replace") as f:
                #         contents = f.read()
                #
                #         soup = BeautifulSoup(contents, 'html.parser')  # lxml
                #         for tr in soup.findAll("tr"):
                #             text = tr.getText()
                #             found_groups = re.findall(regex, text)[0]
                #             gc_1 = "_".join(found_groups[:2])
                #             gc_2 = "_".join(found_groups[2:4])
                #
                #             list_pairs.append({
                #                 'country_code': country_code,
                #                 'gc_1': gc_1,
                #                 'gc_2': gc_2
                #             })
                #     df_samples = pd.concat([df_samples, pd.DataFrame(list_pairs)], ignore_index=True)

    data_test_df.to_csv(color_config.comb_test_filename, index=False)

            # df_samples = df_samples.merge(df_feedback, how='left', on=['gc_1', 'gc_2'])
            # df_samples['compatible'].fillna(1, inplace=True)
            # df_samples['compatible'] = df_samples['compatible'].astype(int)
            # df_samples.to_csv(os.path.join(folder_path, folder, 'data_test.csv'), index=False)
            # print(f"Folder {folder}:")
            # print(df_samples.groupby('country_code').mean()['compatible'])
            # print(f"Global: {np.round(df_samples.groupby('country_code').mean()['compatible'].mean(), 3)}")
    print("Done.")
