from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from flask_wtf import Form
from wtforms.fields import DecimalField, RadioField, StringField, SubmitField
from wtforms.validators import Required, ValidationError

from predict import make_prediction

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret!"


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


@app.route("/home", methods=["POST", "GET"])
def home():
    form = PredictForm()
    # form = YesNoQuestionForm()
    if form.validate_on_submit():
        output = make_prediction(form)
        text = "Predicted price is {}".format(output)
        return render_template("home.html", form=form, prediction_text=text)
    return render_template("home.html", form=form)



if __name__ == "__main__":
    app.run(debug=True)
