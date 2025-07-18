import flask


app = flask.Flask(__name__)

def is_userloggedin():
    return flask.session.get('user_id') is not None

def get_user_id():
    return flask.session.get('user_id', None)