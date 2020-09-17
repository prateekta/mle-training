.. _http://127.0.0.1:5000/: http://www.python.org/
.. |br| raw:: html

   <br />

.. _flask:

Flask Application
============

Web application
----------------

The predictions for the random forest model can be obtained using a simple HTML form 
that allows you to submit the features for your housing price model. To start the application, you can run::

    >>python flask_application/app.py

This will create a localhost server that can be accessed on http://127.0.0.1:5000/. 
|br|
You can enter the values in the fields provided and hit the submit button to get the prediction. 
|br|
|br|
By default, it'll load the application in production mode, which can be altered by going to 
``flask_application/__init__.py`` and changing ``production`` to ``development`` on line 4.

RESTful web API
----------------

There's also an option that takes a json input and returns the predicted price as a json response, 
implemented in the ``\predict`` endpoint inside the application.
|br|
|br|
To use it, you first need to generate credentials, which can be done by running::

    >>python flask_application/generate_cred.py

This will create a file ``flask_application/credentials.txt`` with encrypted credentials. Currently,
the code accepts two pairs of credentials, which can be checked in variable user_database in ``flask_application/app.py``.
|br|
|br|
Once you have the credentials, you can pass the json string using ``curl``, as::

    >>curl -u 'credentials' -i -H "Content-Type: application/json" -X POST -d '{"longitude":2, "latitude":2, "housing_median_age":1000, "total_rooms":2, "total_bedrooms":3, "population":5, "households":4, "median_income":10000, "ocean_proximity":"NEAR BAY"}' http://localhost:5000/predict

where ``credentials`` is the string obtained after running the above code.