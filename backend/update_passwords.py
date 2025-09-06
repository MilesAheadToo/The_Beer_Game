from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from passlib.context import CryptContext

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def update_passwords():
    # Database connection details
    db_user = "beer_user"
    db_password = "Daybreak@2025"
    db_host = "db"
    db_name = "beer_game"
    
    # Create SQLAlchemy engine
    DATABASE_URL = f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}"
    engine = create_engine(DATABASE_URL)
    
    # Create a session
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Hash the new password
        hashed_password = pwd_context.hash("Daybreak@2025")
        
        # Update all users with the new password
        update_query = """
        UPDATE users 
        SET hashed_password = :hashed_password,
            last_password_change = NOW()
        """
        
        session.execute(text(update_query), {"hashed_password": hashed_password})
        session.commit()
        
        # Verify the update
        result = session.execute(text("SELECT username, email FROM users")).fetchall()
        print(f"Updated passwords for {len(result)} users:")
        for row in result:
            print(f"- {row[0]} ({row[1]})")
            
    except Exception as e:
        session.rollback()
        print(f"Error updating passwords: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    update_passwords()
