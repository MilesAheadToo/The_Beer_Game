from alembic.config import Config
from alembic import command
import os

def run_migrations():
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Set the path to the alembic.ini file
    alembic_cfg = os.path.join(script_dir, "..", "alembic.ini")
    
    # Create the config
    config = Config(alembic_cfg)
    
    # Set the script location
    config.set_main_option('script_location', os.path.join(script_dir, "..", "alembic"))
    
    # Run the migration
    command.upgrade(config, 'head')
    print("Database migrations applied successfully!")

if __name__ == "__main__":
    run_migrations()
