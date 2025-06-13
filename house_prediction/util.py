import json
import joblib
import numpy as np
def get_location_names():
    return __locations
__locations = None
__data_columns = None
__model = None

def get_EstimatedPrice(location, size, total_sqft, bath):
    try:
         loc_index=__data_columns.index(location.lower())
    except:
        loc_index = -1
    x = np.zeros(len(__data_columns))
    x[0] = size
    x[1] = total_sqft
    x[2] = bath
    if loc_index >=0:
        x[loc_index]=1
    return round(__model.predict([x])[0],2)


def load_saved_artificats():
    print("Loading saved artifacts...")
    global __data_columns
    global __locations
    global __model
    with open('columns.json','r') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]
    __model=joblib.load('house_price_model.pkl')
    print("Artifacts loaded successfully.")

if __name__ == '__main__':
    load_saved_artificats()
    loc=get_location_names()
    print(len(loc))
    print(get_EstimatedPrice('1st Phase JP Nagar', 5, 1500, 2))
    print(get_EstimatedPrice('Indira Nagar', 3, 1500, 2))
    print(get_EstimatedPrice('Indira Nagar', 4, 1500, 3))
