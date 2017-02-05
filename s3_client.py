import boto
from boto.s3.connection import Location
from boto.s3.key import Key
from boto.exception import S3ResponseError


class s3Client:

    def __init__(self, config):
        self.s3conn = boto.connect_s3(config.get('s3', 'ACCESS_KEY'),
                                      config.get('s3', 'SECRET_KEY'),
                                      host=config.get('s3', 'AWS_HOST'))
        return

    def create_bucket_if_not_exists(self, name, location=Location.USWest2):
        try:
            bucket = self.s3conn.get_bucket(name)
        except S3ResponseError:
            bucket = self.s3conn.create_bucket(name, location=location)

        return bucket

    def upload_to_s3(self, bucket, s3_file_name, local_file):
        k = Key(bucket)
        k.key = s3_file_name
        k.set_contents_from_filename(local_file)

        return

    def download_from_s3(self, bucket, s3_file_name, local_file):
        k = Key(bucket)
        k.key = s3_file_name
        k.get_contents_to_filename(local_file)

        return
