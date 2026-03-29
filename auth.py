from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename
from models import db, User, ProcessingHistory
import re
import os
import secrets
from datetime import datetime, timedelta

auth_bp = Blueprint('auth', __name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
UPLOAD_FOLDER = 'static/profile_photos'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


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


@auth_bp.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    """Handle forgot password request"""
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        
        if not email:
            flash('Please enter your email address', 'error')
        elif not validate_email(email):
            flash('Please enter a valid email address', 'error')
        else:
            user = User.query.filter_by(email=email).first()
            
            if user:
                # Generate a secure token
                token = secrets.token_urlsafe(32)
                user.password_reset_token = token
                user.password_reset_expires = datetime.utcnow() + timedelta(hours=1)
                db.session.commit()
                
                # For basic auth, show the reset link directly
                # In production, this would be sent via email
                reset_url = url_for('auth.reset_password', token=token, _external=True)
                flash(f'Password reset link (valid for 1 hour): Copy this link to reset your password', 'info')
                return render_template('forgot_password.html', reset_link=reset_url, email=email)
            else:
                # Don't reveal if email exists for security
                flash('If an account exists with this email, a password reset link will be provided.', 'info')
    
    return render_template('forgot_password.html')


@auth_bp.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    """Handle password reset with token"""
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    # Find user by token
    user = User.query.filter_by(password_reset_token=token).first()
    
    if not user:
        flash('Invalid or expired reset link', 'error')
        return redirect(url_for('auth.forgot_password'))
    
    # Check if token is expired
    if user.password_reset_expires < datetime.utcnow():
        user.password_reset_token = None
        user.password_reset_expires = None
        db.session.commit()
        flash('Reset link has expired. Please request a new one.', 'error')
        return redirect(url_for('auth.forgot_password'))
    
    if request.method == 'POST':
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        
        if not password:
            flash('Please enter a new password', 'error')
        elif password != confirm_password:
            flash('Passwords do not match', 'error')
        else:
            is_valid, message = validate_password(password)
            if not is_valid:
                flash(message, 'error')
            else:
                # Update password and clear token
                user.set_password(password)
                user.password_reset_token = None
                user.password_reset_expires = None
                db.session.commit()
                
                flash('Password reset successfully! Please log in with your new password.', 'success')
                return redirect(url_for('auth.login'))
    
    return render_template('reset_password.html', token=token)


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
    # Limit to 5 recent activities
    history = current_user.processing_history.order_by(
        ProcessingHistory.created_at.desc()
    ).limit(5).all()
    
    return render_template('profile.html', stats=stats, history=history)


@auth_bp.route('/api/update-username', methods=['POST'])
@login_required
def update_username():
    """Update user's username"""
    try:
        data = request.get_json()
        new_username = data.get('username', '').strip()
        
        if not new_username:
            return jsonify({'error': 'Username cannot be empty'}), 400
        
        if len(new_username) < 3:
            return jsonify({'error': 'Username must be at least 3 characters'}), 400
        
        if len(new_username) > 30:
            return jsonify({'error': 'Username must be less than 30 characters'}), 400
        
        # Check if username already exists (excluding current user)
        existing_user = User.query.filter(
            User.username == new_username,
            User.id != current_user.id
        ).first()
        
        if existing_user:
            return jsonify({'error': 'Username already taken'}), 400
        
        current_user.username = new_username
        db.session.commit()
        
        return jsonify({
            'success': True,
            'username': new_username,
            'message': 'Username updated successfully'
        }), 200
        
    except Exception as e:
        print(f"Error updating username: {e}")
        return jsonify({'error': str(e)}), 500


@auth_bp.route('/api/update-profile-photo', methods=['POST'])
@login_required
def update_profile_photo():
    """Update user's profile photo"""
    try:
        if 'photo' not in request.files:
            return jsonify({'error': 'No photo file provided'}), 400
        
        file = request.files['photo']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg, gif, webp'}), 400
        
        # Create upload folder if not exists
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        
        # Delete old profile photo if exists
        if current_user.profile_photo:
            old_path = os.path.join('static', current_user.profile_photo)
            if os.path.exists(old_path):
                try:
                    os.remove(old_path)
                except:
                    pass
        
        # Save new photo
        filename = secure_filename(f"user_{current_user.id}_{file.filename}")
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Update database
        current_user.profile_photo = f"profile_photos/{filename}"
        db.session.commit()
        
        return jsonify({
            'success': True,
            'photo_url': url_for('static', filename=current_user.profile_photo),
            'message': 'Profile photo updated successfully'
        }), 200
        
    except Exception as e:
        print(f"Error updating profile photo: {e}")
        return jsonify({'error': str(e)}), 500


@auth_bp.route('/api/remove-profile-photo', methods=['POST'])
@login_required
def remove_profile_photo():
    """Remove user's profile photo"""
    try:
        if current_user.profile_photo:
            old_path = os.path.join('static', current_user.profile_photo)
            if os.path.exists(old_path):
                try:
                    os.remove(old_path)
                except:
                    pass
        
        current_user.profile_photo = None
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Profile photo removed'
        }), 200
        
    except Exception as e:
        print(f"Error removing profile photo: {e}")
        return jsonify({'error': str(e)}), 500


@auth_bp.route('/api/change-password', methods=['POST'])
@login_required
def change_password():
    """Change user's password"""
    try:
        data = request.get_json()
        current_password = data.get('current_password', '')
        new_password = data.get('new_password', '')
        
        if not current_password or not new_password:
            return jsonify({'error': 'All fields are required'}), 400
        
        # Verify current password
        if not current_user.check_password(current_password):
            return jsonify({'error': 'Current password is incorrect'}), 400
        
        # Validate new password using same rules as signup
        is_valid, message = validate_password(new_password)
        if not is_valid:
            return jsonify({'error': message}), 400
        
        # Set new password
        current_user.set_password(new_password)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Password changed successfully'
        }), 200
        
    except Exception as e:
        print(f"Error changing password: {e}")
        return jsonify({'error': str(e)}), 500


@auth_bp.route('/api/change-email', methods=['POST'])
@login_required
def change_email():
    """Change user's email address"""
    try:
        data = request.get_json()
        new_email = data.get('new_email', '').strip().lower()
        password = data.get('password', '')
        
        if not new_email or not password:
            return jsonify({'error': 'All fields are required'}), 400
        
        # Verify password
        if not current_user.check_password(password):
            return jsonify({'error': 'Password is incorrect'}), 400
        
        # Validate email format
        email_regex = r'^[^\s@]+@[^\s@]+\.[^\s@]+$'
        if not re.match(email_regex, new_email):
            return jsonify({'error': 'Invalid email format'}), 400
        
        # Check if email already exists
        existing_user = User.query.filter_by(email=new_email).first()
        if existing_user and existing_user.id != current_user.id:
            return jsonify({'error': 'Email already in use'}), 400
        
        # Update email
        current_user.email = new_email
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Email changed successfully'
        }), 200
        
    except Exception as e:
        print(f"Error changing email: {e}")
        return jsonify({'error': str(e)}), 500


@auth_bp.route('/api/delete-account', methods=['POST'])
@login_required
def delete_account():
    """Delete user account and all associated data"""
    try:
        data = request.get_json()
        password = data.get('password', '')
        
        if not password:
            return jsonify({'error': 'Password is required'}), 400
        
        # Verify password
        if not current_user.check_password(password):
            return jsonify({'error': 'Password is incorrect'}), 400
        
        user_id = current_user.id
        
        # Delete user's processing history and associated files
        histories = ProcessingHistory.query.filter_by(user_id=user_id).all()
        for history in histories:
            # Delete associated files
            for path_attr in ['audio_path', 'transcript_path', 'summary_path', 'summary_audio_path']:
                file_path = getattr(history, path_attr, None)
                if file_path and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except:
                        pass
            
            # Delete the history record
            db.session.delete(history)
        
        # Delete profile photo if exists
        if current_user.profile_photo:
            photo_path = os.path.join('static', current_user.profile_photo)
            if os.path.exists(photo_path):
                try:
                    os.remove(photo_path)
                except:
                    pass
        
        # Logout and delete user
        logout_user()
        user = User.query.get(user_id)
        db.session.delete(user)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Account deleted successfully'
        }), 200
        
    except Exception as e:
        print(f"Error deleting account: {e}")
        db.session.rollback()
        return jsonify({'error': str(e)}), 500
