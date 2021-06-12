import databases
import sqlalchemy
from sqlalchemy_utils import database_exists, create_database
from functools import lru_cache
from api import config
from api.models import metadata
from starlette.config import Config


"""
@lru_cache()
def setting():
    return config.Settings()


def database_pgsql_url_config():
    return str(setting().DB_CONNECTION + "://" + setting().DB_USERNAME + ":" + setting().DB_PASSWORD + "@" + setting().DB_HOST + ":" + setting().DB_PORT + "/" + setting().DB_DATABASE)
"""


def database_pgsql_url_config():
    conf = Config(".env")
    return str(conf("DB_CONNECTION") + "://" + conf("DB_USERNAME") + ":" + conf("DB_PASSWORD") + "@" + conf("DB_HOST") + ":" + conf("DB_PORT") + "/" + conf("DB_DATABASE"))


database = databases.Database(database_pgsql_url_config())
engine = sqlalchemy.create_engine(database_pgsql_url_config())

if not database_exists(engine.url):
    create_database(engine.url)

metadata.create_all(engine)
