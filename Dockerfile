#Use Python
FROM python:3.9-slim

#Set the working directory
WORKDIR /app

#Copy the file
COPY requirements.txt ./
COPY app/ ./app/
COPY diabetes.csv .
COPY train.py .


#Install dependcies
RUN pip install --no-cache-dir -r requirements.txt


#Train the model
RUN python train.py

#Expose the API port
EXPOSE 8000

# Copy the rest of the applicatoin code
COPY . .

#Run the Flask application
CMD ["python", "app/app.py"]