from django.db import models
import csv
import io
from django.conf import settings
from django.db import models
from django.db.models.signals import post_save, pre_save
from django.utils.text import slugify
# Create your models here.
class Tests(models.Model):
    pt=[('status','s'),('video','v')]
    ph=[('True','T'),('False','F')]
    pp=[('True','T'),('False','F')]
    sector=models.CharField(max_length=100,null=False,blank=False)
    message=models.CharField(max_length=100,null=False,blank=False)
    # url=models.CharField(max_length=100,null=False,blank=False)
    videolength=models.PositiveIntegerField(null=False,blank=False)
    sharecount=models.PositiveIntegerField(null=False,blank=False)
    commentcount=models.PositiveIntegerField(null=False,blank=False)
    reactioncount=models.PositiveIntegerField(null=False,blank=False)
    post_type=models.CharField(max_length=100,blank=False,choices=pt)
    from_name=models.CharField(max_length=200,blank=True)
    post_hidden=models.BooleanField(max_length=200,blank=True,choices=ph)
    post_pub=models.BooleanField(max_length=200,blank=True,choices=pp)
    link_name=models.CharField(max_length=200,blank=True)
    link_cap=models.CharField(max_length=200,blank=True)
    link_desc=models.TextField(max_length=200,blank=True)
    link_url=models.URLField(max_length=200,blank=True)
    media_id=models.PositiveIntegerField(blank=True)
    parent_id=models.PositiveIntegerField(blank=True)
    media_thumbnail=models.URLField(blank=True)
    media_url=models.URLField(blank=True)
    story_text=models.TextField(blank=True)
    page_id=models.PositiveIntegerField(blank=True)
    created_time=models.TimeField(blank=True)



    # {
    #     "sector": "media",
    #     "message": "hello",
    #     "url": "www.facebook.com",
    #     "videolength": 33,
    #     "sharecount": 100,
    #     "commentcount": 100,
    #     "reactioncount": 100
    # }




from uzair.signals import csv_uploaded
from uzair.validators import csv_file_validator

def upload_csv_file(instance, filename):
    qs = instance.__class__.objects.filter(user=instance.user)
    if qs.exists():
        num_ = qs.last().id + 1
    else:
        num_ = 1
    return f'csv/{num_}/{instance.user.username}/{filename}'

class CSVUpload(models.Model):
    user        = models.ForeignKey(settings.AUTH_USER_MODEL,on_delete=models.CASCADE)
    file        = models.FileField(upload_to=upload_csv_file, validators=[csv_file_validator])
    completed   = models.BooleanField(default=False)

    def __str__(self):
        return self.user.username


def convert_header(csvHeader):
    header_ = csvHeader[0]
    cols = [x.replace(' ', '_').lower() for x in header_.split(",")]
    return cols


def csv_upload_post_save(sender, instance, created, *args, **kwargs):
    if not instance.completed:
        csv_file = instance.file
        decoded_file = csv_file.read().decode('utf-8')
        io_string = io.StringIO(decoded_file)
        reader = csv.reader(io_string, delimiter=';', quotechar='|')
        header_ = next(reader)
        header_cols = convert_header(header_)
        parsed_items = []

        '''
        if using a custom signal
        '''
        for line in reader:
            parsed_row_data = {}
            i = 0
            row_item = line[0].split(',')
            for item in row_item:
                key = header_cols[i]
                parsed_row_data[key] = item
                i+=1
            parsed_items.append(parsed_row_data)
        csv_uploaded.send(sender=instance, user=instance.user, csv_file_list=parsed_items)
        ''' 
        if using a model directly
        for line in reader:
            new_obj = YourModelKlass()
            i = 0
            row_item = line[0].split(',')
            for item in row_item:
                key = header_cols[i]
                setattr(new_obj, key) = item
                i+=1
            new_obj.save()
        '''
        instance.completed = True
        instance.save()


post_save.connect(csv_upload_post_save, sender=CSVUpload)