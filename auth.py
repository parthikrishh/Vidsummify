from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required, current_user
from models import db, User, ProcessingHistory
import re

auth_bp = Blueprint('auth', __name__)


def validate_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def validate_password(password):
    """Validate password strength"""
    if len(password) < 6:
        return False, "Password must be at least 6 characters long"
    if not any(char.isdigit() for char in password):
        return False, "Password must contain at least one digit"
    if not any(char.isalpha() for char in password):
        return False, "Password must contain at least one letter"
    return True, "Valid"


@auth_bp.route('/signup', methods=['GET', 'POST'])
def signup():
    """User registration"""
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        
        # Validation
        if not username or len(username) < 3:
            flash('Username must be at least 3 characters long', 'error')
        elif not validate_email(email):
            flash('Invalid email address', 'error')
        elif password != confirm_password:
            flash('Passwords do not match', 'error')
        else:
            is_valid, message = validate_password(password)
            if not is_valid:
                flash(message, 'error')
            else:
                # Check if user already exists
                if User.query.filter_by(username=username).first():
                    flash('Username already exists', 'error')
                elif User.query.filter_by(email=email).first():
                    flash('Email already registered', 'error')
                else:
                    # Create new user
                    user = User(username=username, email=email)
                    user.set_password(password)
                    db.session.add(user)
                    db.session.commit()
                    flash('Account created successfully! Please log in.', 'success')
                    return redirect(url_for('auth.login'))
    
    return render_template('signup.html')


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        username_or_email = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        # Find user by username or email
        user = User.query.filter_by(username=username_or_email).first()
        if not user:
            user = User.query.filter_by(email=username_or_email).first()
        
        if user and user.check_password(password):
            login_user(user)
            # Update last login
            user.last_login = db.func.now()
            db.session.commit()
            flash('Logged in successfully!', 'success')
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('home'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')


@auth_bp.route('/logout')
@login_required
def logout():
    """User logout"""
    logout_user()
    flash('Logged out successfully!', 'success')
    return redirect(url_for('auth.login'))


@auth_bp.route('/profile', methods=['GET'])
@login_required
def profile():
    """User profile page"""
    stats = current_user.get_stats()
    history = current_user.processing_history.order_by(
        ProcessingHistory.created_at.desc()
    ).limit(10).all()
    
    return render_template('profile.html', stats=stats, history=history)
