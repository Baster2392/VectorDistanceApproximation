import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
from training_and_searching.sample_based_distance.eb_training import *

from models.siamese_model import SiameseNetwork

from sklearn.model_selection import train_test_split

CSV_FILE_PATH = '../../../../saved_results/res2.csv'

def filter_multiple_zeros(df):
  # Count the number of zeros in each row
  zero_counts = df.eq(0).sum(axis=1)
  # Filter the dataframe to rows with more than one zero
  filtered_df = df[zero_counts > 1]
  return filtered_df







if __name__ == '__main__':
    dataset = []

    df = pd.read_csv('../databases/movie_metadata.csv', skiprows=1, header=None, nrows=3000)

    # Check columns if headers are not included
    print("DataFrame head to inspect column indices and data:")
    print(df.head())


    # List of columns to ignore (like URLs and descriptions)
    columns_to_ignore = [0, 1, 4, 6, 9, 10, 11, 14, 15, 16, 17, 19, 20, 21]

    # Drop the columns to ignore
    df_filtered = df.drop(df.columns[columns_to_ignore], axis=1)

    # Fill NaN values with 0
    df_filtered.fillna(0, inplace=True)




    # Identify string columns that need encoding
    string_columns = df_filtered.select_dtypes(include=['object']).columns

    # Convert remaining string columns to numerical values
    column_mappings = {}
    for column in string_columns:
        unique_values = df_filtered[column].unique()
        mapping = {value: idx for idx, value in enumerate(unique_values)}
        df_filtered[column] = df_filtered[column].map(mapping)
        column_mappings[column] = mapping

    # Replace remaining NaN values with 0 after conversions
    df_filtered.fillna(0, inplace=True)

    df_filtered = filter_multiple_zeros(df_filtered.copy())

    # Convert DataFrame to a list of lists
    data_list = df_filtered.values.tolist()


    # Let's specify the index of the target column in the dataset
    target_column_index = 11

    # Extract features and labels from the dataset
    features = [item[:target_column_index] + item[target_column_index + 1:] for item in data_list]
    labels = [item[target_column_index] for item in data_list]

    # Divide dataset into training and test data with stratification
    X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.2)

    similarity_database = {}

    for i in range(len(X_train)):
        for j in range(i + 1, len(X_train)):
            difference = abs(Y_train[i] - Y_train[j])
            similarity_database[(tuple(X_train[i]), tuple(X_train[j]))] = difference

    similarity_database_test = {}

    for i in range(len(X_test)):
        for j in range(i + 1, len(X_test)):
            difference = abs(Y_test[i] - Y_test[j])
            similarity_database_test[(tuple(X_test[i]), tuple(X_test[j]))] = difference

    #traing the model with the similarity database/ test dataset

    avg_distance = calculate_average_distance(similarity_database)
    print("Average Distance in Similarity Database:", avg_distance)


    avg_distance2 = calculate_average_distance(similarity_database_test)
    print("Average Distance in Test Similarity Database:", avg_distance2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SiameseNetwork(13, 500, 10, 5)
    criterion = nn.L1Loss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=100, factor=0.46, min_lr=1e-10, verbose=True)

    model, _, _, _ = train(model, custom_loss, optimizer, scheduler, epochs=100000, n_samples=300,
                           similarity_database=similarity_database, loss_tolerance=0.4, device=device)

    # Update similarity database using the trained model
 #   similarity_database = update_similarity_database(model, X_train, dataset, similarity_database)

    print(f'=====================================================')
    print(f'=====================================================')


    # Train the model
 #   model, _, _, _ = train(model, custom_loss, optimizer, scheduler, epochs=100000, n_samples=200,
  #                         similarity_database=similarity_database, loss_tolerance=1, device=device)



    x_validate, y_validate = generate_sample_data(1000, 0, 100000, model.input_dim, similarity_database_test, False)
    x_validate = torch.tensor(x_validate, dtype=torch.float).to(device)
    y_validate = torch.tensor(y_validate, dtype=torch.float).to(device)



    validate(model, criterion, x_validate, y_validate)


