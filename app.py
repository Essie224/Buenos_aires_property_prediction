import streamlit as st
import pandas as pd
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os



# Set page config
st.set_page_config(page_title="Buenos Aires Real Estate Dashboard", page_icon ="🌆", layout="wide")


# load clean dataset
df = pd.read_csv("buenos_aires_real_estate_data.csv")

# load trained model
model = joblib.load('house_price_model_1.pkl')


# Create storage directory and file if it doesn't exist
LOG_FILE = 'new_data/predictions.csv'
os.makedirs("new_data", exist_ok=True)
if not os.path.exists(LOG_FILE):
  pd.DataFrame(columns=[
        'property_type', 'surface_covered_in_m2', 'city', 'lat', 'lon',
        'prediction'
  ]).to_csv(LOG_FILE, index=False)


    # Helper: Log prediction
def log_prediction(input_data, prediction):
    # Decode encoded values before saving
    reverse_property_type_map = {v: k for k, v in property_type_map.items()}
    reverse_city_map = {v: k for k, v in city_map.items()}

    input_data['property_type'] = reverse_property_type_map.get(input_data['property_type'], input_data['property_type'])
    input_data['city'] = reverse_city_map.get(input_data['city'], input_data['city'])
    input_data['prediction'] = prediction
    df = pd.read_csv(LOG_FILE)
    df = pd.concat([df, pd.DataFrame([input_data])], ignore_index=True)
    df.to_csv(LOG_FILE, index=False)



    # Helper: Make prediction
def make_prediction(new_data):
    df = pd.DataFrame(new_data, index=[0])
    prediction = model.predict(df)[0]
    log_prediction(new_data, float(prediction))
    return prediction

# app title
st.title("🌆 Buenos Aires Real Estate Prediction App")
st.markdown("***Welcome to the interactive property dashboard!***")
st.markdown("#### Explore data, visualize insights and predict property prices 🏘")

# Create tabs
tabs = st.tabs(["📝 Introduction", "📊 Data Exploration", "🤖 Prediction Model", "📁 Collected Data"])
  

# Tab 1: Introduction
with tabs[0]:
    st.header("Project Overview")
    st.markdown("""
    ### 📌 Problem Statement
    Buenos Aires Real Estate Analysis is a data-driven project aimed at understanding the property market in Buenos Aires, Argentina.
     
    ### 🎯 Project Objective
    This Project used a combination of data exploration, visualization, and machine learning to 
    - explore patterns in property prices,
    - identify key factors that influence property value,
    - provide actionable insights for buyers, sellers, and investors.
    """)

    ### 📚 Dataset
    try:
      st.dataframe(df.head())
    except FileNotFoundError:
        st.warning("⚠️ df not found")

    
# Tab 2: Data Exploration
with tabs[1]:
    st.header("📊 Data Exploration")
    st.markdown("Here are some visualizations showing feature relationships with prices:")

    for filename, caption in [
        ("PRICE VS SIZE.png", "PRICE VS SIZE"),
        ("AVG PRICE BY PROPERTY TYPE.png", "AVG PRICE BY PROPERTY TYPE"),
        ("AVG PRICE BY CITY.png", "AVG PRICE BY CITY"),
        ("PRICE VS LOCATION.png", "PRICE VS LOCATION"),
        ("FEATURE CORRELATION HEATMAP.png", "FEATURE CORRELATION HEATMAP")
    ]:
        try:
            img = Image.open(f"assets/images/{filename}")
            st.subheader(f"📈 {caption}")
            st.image(img)
        except FileNotFoundError:
            st.warning(f"⚠️ Missing image: assets/{filename}")
        except Exception as e:
            st.error(f"Error loading image: {e}")


         # Add model evaluation and result as an image
    st.markdown("## 📄 Model Evaluation Results")
    try:
      eval_df = pd.read_csv("model_evaluation_results.csv")
      st.dataframe(eval_df)
      
      st.markdown("### 🧠 Interpretation of Results")
      st.markdown("""
      - **Price vs size**: Larger properties tend to have higher prices however, some small properties,
        still appear expensive maybe because of high-end features or location.
      - **Price vs Property type**: On average **houses** tend to be priced higher than other properties(e.g Apartment, store etc)
      - **Price vs city**: Cities like **Catalinas**, **Puerto Madero** and **Recoleta** dominate,
        the top of the price chart by average.
      - **Price vs location**: high value properties cluster around **central and coastal areas**,
        the closer a property is to the central the higher the price.
      - **Correlation heatmap**: Price is *strongly correlated* with surface_total_in_m2,  
        confirming that *size* is a major price driver.  

        🏆 The model generalizes well and is suitable for real-world prediction tasks.
        """)
    except FileNotFoundError:
      st.warning("⚠️ model  evaluation image not found.")



# Tab 4: Prediction Model
with tabs[2]:
    st.header("🤖 Predict Property Prices")

    st.write("Fill in the details below to get an estimated property price in Buenos Aires:")

    property_type_map = {'PH': 0, 'apartment': 1, 'house': 2, 'store': 3}
    property_type_name = st.selectbox("Property Type", list(property_type_map.keys()), help="Select the type of property.")
    property_type = property_type_map[property_type_name]

    surface_covered_in_m2 = float(st.number_input(
    "Total Surface Area (m²)",
    min_value=10.0,
    max_value=1000.0,
    value=80.0,
    help="Enter the total surface area in square meters (e.g., 80 for an average apartment)."
    ))

    

    city_map = {'': 0, 'Abasto': 1, 'Agronomía': 2, 'Almagro': 3, 'Almirante Brown': 4, 'Avellaneda': 5,
       'Balvanera': 6, 'Barracas': 7, 'Barrio Norte': 8, 'Belgrano': 9, 'Berazategui': 10, 'Boca': 11, 
       'Boedo': 12, 'Caballito': 13, 'Catalinas': 14, 'Cañuelas': 15, 'Centro / Microcentro': 16,
       'Chacarita': 17, 'Coghlan': 18, 'Colegiales': 19, 'Congreso': 20, 'Constitución': 21, 
       'Escobar': 22, 'Esteban Echeverría': 23, 'Ezeiza': 24, 'Florencio Varela': 25, 'Flores': 26, 
       'Floresta': 27, 'General Rodríguez': 28, 'General San Martín': 29, 'Hurlingham': 30, 
       'Ituzaingó': 31, 'José C Paz': 32, 'La Matanza': 33, 'La Plata': 34, 'Lanús': 35, 
       'Las Cañitas': 36, 'Liniers': 37, 'Lomas de Zamora': 38, 'Malvinas Argentinas': 39, 
       'Marcos Paz': 40, 'Mataderos': 41, 'Merlo': 42, 'Monserrat': 43, 'Monte Castro': 44, 
       'Moreno': 45, 'Morón': 46, 'Nuñez': 47, 'Once': 48, 'Palermo': 49, 'Parque Avellaneda': 50, 
       'Parque Centenario': 51, 'Parque Chacabuco': 52, 'Parque Chas': 53, 'Parque Patricios': 54, 
       'Paternal': 55, 'Pilar': 56, 'Pompeya': 57, 'Presidente Perón': 58, 'Puerto Madero': 59, 
       'Quilmes': 60, 'Recoleta': 61, 'Retiro': 62, 'Saavedra': 63, 'San Cristobal': 64, 
       'San Fernando': 65, 'San Isidro': 66, 'San Miguel': 67, 'San Nicolás': 68, 'San Telmo': 69, 
       'San Vicente': 70, 'Tigre': 71, 'Tres de Febrero': 72, 'Tribunales': 73, 'Velez Sarsfield': 74, 
       'Versalles': 75, 'Vicente López': 76, 'Villa Crespo': 77, 'Villa Devoto': 78, 
       'Villa General Mitre': 79, 'Villa Lugano': 80, 'Villa Luro': 81, 'Villa Ortuzar': 82, 
       'Villa Pueyrredón': 83, 'Villa Real': 84, 'Villa Santa Rita': 85, 'Villa Soldati': 86, 
       'Villa Urquiza': 87, 'Villa del Parque': 88}

    city_name = st.selectbox("City", list(city_map.keys()))
    city = city_map[city_name]
    
    lat = float(st.number_input(
    "Latitude",
    value=-34.6,
    format="%.6f",
    help="Enter the latitude of the property (e.g., -34.6 for Buenos Aires)."
    ))

    lon = float(st.number_input(
    "Longitude",
    value=-58.4,
    format="%.6f",
    help="Enter the longitude of the property (e.g., -58.4 for Buenos Aires)."
    ))



    # Prepare input for model ( 2d array)
    if st.button("🧠 Predict Price"):
      input_data = {
      "property_type": property_type,
      "surface_covered_in_m2": surface_covered_in_m2,
      "city": city,
      "lat": lat,
      "lon": lon
    }

      pred = make_prediction(input_data)
      st.success(f"💰 Estimated Property Price: ${pred:,.2f}")




with tabs[3]:
  st.header("📁 Collected Prediction Data")
  try:
    df = pd.read_csv(LOG_FILE)
    st.dataframe(df)
    st.download_button(
    label="Download CSV",
    data=df.to_csv(index=False),
    file_name="user_predictions.csv",
    mime="text/csv"
        )
  except Exception as e:
    st.error(f"Error loading prediction log: {e}")
      
     
    
# # Predict button
# if st.button("Predict Price"):
#   prediction = model.predict(input_data)
#   st.success(f"💰 Estimated Property Price: ${prediction[0]:,.2f}")




# 
# # Tab 3: Insights and report
# with tabs[3]:
#   st.title("🔎Buenos Aires Real Estate")
#   st.markdown("""
#   This section summarizes the main findings from the Buenos Aires real-estate dataset and the predictive model.
#   """)

# # Executive summary 
#   st.header("Executive Summary")
#   st.markdown("""
#   This project analyzed housing data from Buenos Aires to understand property prices and build a 
#   predictive model. After cleaning and exploring the dataset, we developed a Random Forest 
#   model that reduced prediction error by nearly 55% compared to a simple baseline guess.

#   *Finding:* Location, city, property size, and property type are the most important factors influencing price.
#   The model provides quick price estimates to support buyers, sellers, and investors when making decisions.
#   """)

# # Data snapshot (if df exists)
#   try:
#     st.header("Data Snapshot")
#     st.write("Below is a quick look at the cleaned dataset used in this analysis.")
#     st.dataframe(df.head())
#     st.markdown(f"- *Rows:* {df.shape[0]}  \n- *Columns:* {df.shape[1]}")
#   except Exception:
#     st.info("DataFrame df not found in the app environment. Make sure df = pd.read_csv(...) is defined earlier in main.py.")

# # Key findings
#   st.header("Key Findings & Evidence")

#   st.markdown("### 1) Location  affects price")
#   st.markdown("""
#   Properties in central and coastal city tend to be pricier.  
#   (Visual evidence: the location-price map and the city bar charts available in the exploration tab.)  \n
#   **Takeaway:**  
#   For sellers : listing near high-demand city helps fetch higher offers. 
#   For buyers : consider city slightly further out for better value.
#   """)

# # Price vs Surface chart
# st.subheader("Price vs Surface Area")
# st.markdown("Larger properties generally command higher prices. Below is a compact scatter with a trendline.")

# try:
#   fig, ax = plt.subplots(figsize=(7, 4))
#   sns.regplot(x="surface_covered_in_m2", y="price",
#                 data=df.sample(min(1000, max(100, df.shape[0]))), scatter_kws={"s": 10, "alpha": 0.6}, ax=ax)
#   ax.set_xlabel("Surface covered (m²)")
#   ax.set_ylabel("Price (USD)")
#   ax.set_title("Price vs Surface Area — sample of listings")
#   st.pyplot(fig)
# except Exception:
#   st.info("Price vs Surface plot could not be rendered — check that df contains surface_total and price (or price_usd).")

# st.markdown("*Takeaway:* Size matters, a bigger surface area is associated with higher price, though some small properties, still appear expensive.")

#  # Average price by property type
# st.subheader("Average Price by Property Type")
# st.markdown("This chart shows how average prices differ by property type (apartment, house, etc.).")

# try:
#   type_col = "property_type"
#   price_col = "price"
#   avg_price_by_type = df.groupby(type_col)[price_col].mean().sort_values(ascending=False)
#   st.bar_chart(avg_price_by_type)
#   st.markdown("**Takeaway:** Houses tend to have higher average prices than apartments of similar size — often due to land and privacy.")
# except Exception:
#   st.info("Could not compute average price by property type. Ensure your DataFrame has columns named property_type and price (or price_usd).")

#  # Correlation heatmap
# st.subheader("Correlation between numeric features")
# st.markdown("A heatmap shows which numeric features move together. Focus on correlations with price (positive or negative).")

# try:
#   numeric_df = df.select_dtypes(include=["number"])
#   corr = numeric_df.corr()
#   fig2, ax2 = plt.subplots(figsize=(9, 7))
#   sns.heatmap(corr, annot=True, fmt=".2f", ax=ax2)
#   ax2.set_title("Feature correlation matrix")
#   st.pyplot(fig2)
#   st.markdown("**Takeaway:** Look for strong positive correlations with price (e.g., surface_covered) and negative correlations if present.")
# except Exception:
#   st.info("Correlation heatmap could not be produced. Ensure df has numeric columns like surface_total, lat, lon, price.")


# st.header("Model Performance")
# st.markdown("""
# Below show the model evaluation metrics (MAE, RMSE, R²) to give an idea of prediction accuracy.  
# MAE = average absolute error in the same units as price (lower is better).  
# R² = how well the model performs on seen data (the higher the better).
# """)

# try:
#   eval_df = pd.read_csv("model_evaluation_results.csv")
#   st.dataframe(eval_df)
# except Exception:
#   st.info("Model evaluation table not found as model_evaluation_result.csv.")


# # Feature importance
# st.header("Model Features?")
# st.markdown("""
#  features: **surface area**, **city**, **lat**, **lon**, and **property type**.
#  """)


# # Recommendations
# st.header(" Recommendations")
# st.markdown("""
# - *For buyers:* Focus on neighborhoods with better price-to-size ratios, So instead of paying a lot,
#    for a small apartment in a trendy area, you might find a nearby neighborhood where the same money, 
#    buys you a larger home or more land, you can often find good value slightly outside the most central areas.  
# - *For sellers:* Highlight property size and nearby facilities in listings, since these increase buyer,
#    interest. Even small improvements, like parking, a balcony, or good maintenance, can support higher,
#   asking prices.  
# - *For investors:* Consider areas with steady price growth and rental demand; use the model to estimate fair purchase prices before bidding.
# """)






# st.dataframe(df.head())

# st.sidebar.success("Select a page above to navigate  👆")