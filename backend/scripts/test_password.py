from passlib.context import CryptContext

# The hash from the database
hashed_password = "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW"

# Create a password context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Test the password
print("Testing password 'password':", pwd_context.verify("password", hashed_password))
