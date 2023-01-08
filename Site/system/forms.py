from .models import Unit
from django.forms import ModelForm, NumberInput

class UnitForm(ModelForm):
	class Meta:
		model = Unit
		fields = ['user_id']
		
		widgets = {
			"user_id": NumberInput(attrs={
				'class': 'form-control',
				'placeholder': 'user_id'
			}),		
		}
