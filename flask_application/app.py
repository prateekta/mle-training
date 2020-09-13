import base64

from flask import (
    Flask,
    abort,
    flash,
    jsonify,
    make_response,
    redirect,
    render_template,
    request,
    session,
    url_for
)
from flask_login import LoginManager, UserMixin, login_required, login_user
from flask_wtf import Form
from itsdangerous import JSONWebSignatureSerializer
from wtforms.fields import DecimalField, StringField, SubmitField
from wtforms.validators import Required, ValidationError

from flask_application import app
from predict import make_prediction

login_manager = LoginManager()
login_manager.init_app(app)


class User(UserMixin):
    # proxy for a database of users
    user_database = {
        "prateek": ("prateek", "flask"),
        "test": ("test", "password"),
    }

    def __init__(self, username, password):
        self.id = username
        self.password = password

    @classmethod
    def get(cls, id):
        return cls.user_database.get(id)


@login_manager.request_loader
def load_user(request):
    token = request.headers.get("Authorization")
    if token is None:
        token = request.args.get("token")

    if token is not None:
        token = token.replace("Basic ", "", 1)
        try:
            token = base64.b64decode(token).decode("utf-8")
        except TypeError:
            pass
        #print("the token is : ", token, app.secret_key)
        s = JSONWebSignatureSerializer(app.secret_key)
        # token = s.loads(token)
        username, password = token.split(":")  # naive token
        username, password = s.loads(username), s.loads(password)
        user_entry = User.get(username)
        # print("user_entry", user_entry)
        if user_entry is not None:
            user = User(user_entry[0], user_entry[1])
            # print(user.password)
            if user.password == password:
                return user
    return None


def ocean_proximity_check(form, field):
    if str(field.data).lower() not in [
        "inland",
        "island",
        "near bay",
        "near ocean",
    ]:
        raise ValidationError(
            "Field must be one of 'inland','island','near bay', 'near ocean'"
        )


class PredictForm(Form):
    longitude = DecimalField(
        "Longitude:", validators=[Required(message="should be a float value.")]
    )
    latitude = DecimalField(
        "Latitude:", validators=[Required(message="should be a float value.")]
    )
    housing_median_age = DecimalField(
        "Housing Median Age:",
        validators=[Required(message="should be a float value.")],
    )
    total_rooms = DecimalField(
        "Total Rooms:",
        validators=[Required(message="should be a float value.")],
    )
    total_bedrooms = DecimalField(
        "Total Bedrooms:",
        validators=[Required(message="should be a float value.")],
    )
    population = DecimalField(
        "Population:",
        validators=[Required(message="should be a float value.")],
    )
    households = DecimalField(
        "Households:",
        validators=[Required(message="should be a float value.")],
    )
    median_income = DecimalField(
        "Median Income:",
        validators=[Required(message="should be a float value.")],
    )
    ocean_proximity = StringField(
        "Ocean Proximity:", validators=[Required(), ocean_proximity_check]
    )
    submit = SubmitField("Submit")


def validate_json(data):
    try:
        assert isinstance(data["latitude"], int) or isinstance(
            data["latitude"], float
        )
        assert isinstance(data["longitude"], int) or isinstance(
            data["longitude"], float
        )
        assert isinstance(data["housing_median_age"], int) or isinstance(
            data["housing_median_age"], float
        )
        assert isinstance(data["total_rooms"], int)
        assert isinstance(data["population"], int)
        assert isinstance(data["total_bedrooms"], int)
        assert isinstance(data["households"], int)
        assert isinstance(data["median_income"], int) or isinstance(
            data["median_income"], float
        )
        assert str(data["ocean_proximity"]).lower() in [
            "inland",
            "island",
            "near bay",
            "near ocean",
        ]
        return True
    except AssertionError:
        return False


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({"error": "Not found"}), 404)


@app.errorhandler(422)
def wrong_data(error):
    return make_response(jsonify({"error": "wrong data"}), 422)


@app.route("/home", methods=["POST", "GET"])
@app.route("/", methods=["POST", "GET"])
def home():
    form = PredictForm()
    # form = YesNoQuestionForm()
    if form.validate_on_submit():
        flash("making prediction for the data given...")
        output = make_prediction({}, form)
        text = "Predicted price is {}".format(output)
        return render_template("home.html", form=form, prediction_text=text)
    return render_template("home.html", form=form)


@app.route("/predict", methods=["POST"])
@login_required
def results():

    if not request.json:
        abort(400)

    data = request.get_json(force=True)
    if not validate_json(data):
        abort(422)
    prediction = make_prediction(data)

    return jsonify(prediction)


if __name__ == "__main__":
    app.run()


# s = JSONWebSignatureSerializer('app.secret_key')
# s.dumps('prateek')
# s.dumps('flask')
# curl -u 'eyJhbGciOiJIUzUxMiJ9.InByYXRlZWsi.6pTTZjSXEyyqq_RMPaM53H9B-GMaT7sBZyPucNm-agpuSh4YY6573lUGwMTsTiGHsyuqN9MOKS9F6xWFK_kDYg':'eyJhbGciOiJIUzUxMiJ9.ImZsYXNrIg._s8ubXhQqH_s3RfO4CrPL5keU_s04k-1ZefmdIxtSS3m_aJsY9asSNpZDISjp_hVpvJLEnkislqe42enl8qtnQ' -i -H "Content-Type: application/json" -X POST -d '{"longitude":2, "latitude":2, "housing_median_age":1000, "total_rooms":2, "total_bedrooms":3, "population":5, "households":4, "median_income":10000, "ocean_proximity":"NEAR BAY"}' http://localhost:5000/predict
