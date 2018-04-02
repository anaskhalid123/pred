# Generated by Django 2.0.3 on 2018-04-01 12:28

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Test',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('sector', models.CharField(max_length=100)),
                ('message', models.CharField(max_length=100)),
                ('url', models.CharField(max_length=100)),
                ('videolength', models.PositiveIntegerField()),
                ('sharecount', models.PositiveIntegerField()),
                ('commentcount', models.PositiveIntegerField()),
                ('reactioncount', models.PositiveIntegerField()),
            ],
        ),
    ]
