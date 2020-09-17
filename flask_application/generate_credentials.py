from itsdangerous import JSONWebSignatureSerializer

from flask_application import app

s = JSONWebSignatureSerializer(app.secret_key)

user = s.dumps("prateek")
password = s.dumps("flask")

with open("credentials.txt", "w+") as f:
    f.write(":".join([user.decode("utf-8"), password.decode("utf-8")]))
