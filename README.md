# dublin-bikes

This is a Python implementation to fulfill BSc. Computer Science and Software Engineering through Science (Accelerated) course at Maynooth University. The goal of the project is to understand how to pre-processing data, identity usage patterns of stations and using the Gradient Boosting algorithm to predict the free stands as well as the available bikes at a specific station at a short period of time.

### Data source
Data was obtained from JCDecaux API every 5 minutes via a server located in Maynooth University. It was stored in CSV and JSON files and are available on [here](https://drive.google.com/drive/folders/1-aKHTT0b3yw1T49lE71EhrKFePPSek8q?usp=sharing). There are more than 120,000 JSON files and 100 CSV files

### How to run the project
The Python vesrion should be 3.7.0 or higher
> $ python db-new-data-preparation.py

> $ python db-all-exploration.py

> $ python db-clustering.py

> $ python db-modeling.py

> $ python db-api.py

### The application of the built predicting model
After the model was implemented, a web-based mobile application was built to make use of the model. That application located in ./DublineBikesForecast is called Dublin Bikes Forecast. Please see README.md inside that directory for running the application.

### References
1. An exploratory analysis on Dublin Bikes data was introduced when the bike scheme is in the inital stage with 40 stations and 450 bicycles

Mural.maynoothuniversity.ie. (2014). Preliminary Results of a Spatial Analysis of Dublin City’s Bike Rental Scheme - MURAL - Maynooth University Research Archive Library. [online] Available at: http://mural.maynoothuniversity.ie/4919

2. Another study was conducted to find and analyze the busiest and quietest stations when the scheme was expanded to have 101 stations

Pham Thi, Thanh Thoa & Timoney, Joe & Ravichandran, Shyram & Mooney, Peter & Winstanley, A. (2017). Bike Renting Data Analysis: The Case of Dublin City. [online] Available at: https://www.researchgate.net/publication/316451461_Bike_Renting_Data_Analysis_The_Case_of_Dublin_City

3. Two recent thesises of other students on prediction modeling implementation:

Carlos Amaral: https://github.com/amaralcs/dublin_bikes

Nidhin George: https://github.com/nidhgeo/dublinbikes
