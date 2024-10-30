<template>
  <Card style="width: 35rem; overflow: scroll">
    <template #header>
      <img alt="user header" src="../assets/sd_header.jpg" />
    </template>
    <template #title> <b>Skin Disease Classifier</b> </template>
    <template #content>
      <ConfirmDialog style="width: 30rem; overflow: auto">
        <template #message>
          <div>
            <p><strong>Condition: </strong>{{ prediction }}.</p>
            <p><strong>Discription: </strong>{{ discription }} </p>
            <p><strong>Suggestion: </strong>{{ suggestion }}</p>
          </div>
        </template>
      </ConfirmDialog>
      <FileUpload
        name="demo[]"
        :customUpload="true"
        @uploader="myUploader"
        :maxFileSize="1000000"
        accept="image/*"
        :multiple="false"
      >
        <template #empty>
          <span>Drag and drop files to here to upload.</span>
        </template>
    </FileUpload>
    </template>
  </Card>
</template>

<script>
import FileUpload from 'primevue/fileupload';
import ConfirmDialog from 'primevue/confirmdialog';
import Card from 'primevue/card';
export default {
  components: {
    FileUpload,
    Card,
    ConfirmDialog,
  },
  data() {
    return {
      prediction: '',
      discription: '',
      suggestion: '',
    };
  },
  methods: {
    async myUploader(event) {
      console.log('upload img')

      const file = event.files[0];
      const formData = new FormData();
      formData.append('image', file);
      this.$axios.post('http://localhost:8000/api/images/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      }).then(response => {
        console.log('upload success', response.data);
        this.displayResult(response.data)
      }).catch(error => {
        console.error('upload fail', error);
        this.displayResult('Sorry, something went wrong.')
      });
    },
    displayResult(result)
    {
      console.log('display result')
      this.prediction = result['prediction'];
      this.discription = result['description'];
      this.suggestion = result['suggestion'];
      this.$confirm.require({
        header: 'Detection Result',
        icon: 'pi pi-info-circle',
        acceptProps: {
          label: 'Correct',
          severity: 'success',
          icon: 'pi pi-check',
        },
        rejectProps: {
          label: 'Wrong',
          severity: 'danger',
          icon: 'pi pi-times',
        },
        accept: () => {
          console.log('ok')
        },
        reject: () => {
          console.log('oh no')
        }
      });
    },
  }
};
</script>

<style>
  p {
    overflow-wrap: break-word;
  }
</style>