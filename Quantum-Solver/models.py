from datetime import datetime
from app import db
from flask_dance.consumer.storage.sqla import OAuthConsumerMixin
from flask_login import UserMixin
from sqlalchemy import UniqueConstraint
import json


class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.String, primary_key=True)
    email = db.Column(db.String, unique=True, nullable=True)
    first_name = db.Column(db.String, nullable=True)
    last_name = db.Column(db.String, nullable=True)
    profile_image_url = db.Column(db.String, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    
    solved_problems = db.relationship('SolvedProblem', backref='user', lazy=True, cascade='all, delete-orphan')


class OAuth(OAuthConsumerMixin, db.Model):
    user_id = db.Column(db.String, db.ForeignKey(User.id))
    browser_session_key = db.Column(db.String, nullable=False)
    user = db.relationship(User)

    __table_args__ = (UniqueConstraint(
        'user_id',
        'browser_session_key',
        'provider',
        name='uq_user_browser_session_key_provider',
    ),)


class SolvedProblem(db.Model):
    __tablename__ = 'solved_problems'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String, db.ForeignKey('users.id'), nullable=False)
    equation_type = db.Column(db.String(100), nullable=False)
    equation_name = db.Column(db.String(200), nullable=False)
    parameters = db.Column(db.Text, nullable=False)
    solution = db.Column(db.Text, nullable=False)
    steps = db.Column(db.Text, nullable=False)
    graph_data = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.now)
    
    def get_parameters(self):
        return json.loads(self.parameters)
    
    def set_parameters(self, params):
        self.parameters = json.dumps(params)
    
    def get_solution(self):
        return json.loads(self.solution)
    
    def set_solution(self, sol):
        self.solution = json.dumps(sol)
    
    def get_steps(self):
        return json.loads(self.steps)
    
    def set_steps(self, steps_list):
        self.steps = json.dumps(steps_list)
