<template>
  <div class="home container p-3">
    <div class="row">
      <div class="col-12 d-flex flex-column align-items-start">
        <div class="w-50">
          <p>Загрузите снимок</p>
          <b-form-file
            v-model="image"
            placeholder="Выберите или перетащите файл"
            drop-placeholder="Перетащите файл сюда"
            browse-text="Загрузить"
          ></b-form-file>
        </div>
        <div class="d-flex flex-row align-items-center mt-3">
          <b-button variant="primary" @click="submit">{{ isLoading ? 'Обработка...' : 'Отправить' }}</b-button>
          <span v-if="errorMessage" class="text-danger ml-2">{{errorMessage}}</span>
        </div>
        <div v-if="diagnosis" class="mt-4">
          <p>{{ diagnosis }}</p>
          <img :src="imagePreview" height="700px" width="auto"/>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import Vue from 'vue';
import axios from 'axios';
import { BFormFile } from 'bootstrap-vue';
Vue.component('b-form-file', BFormFile);

export default {
  name: "HomeView",
  data() {
    return {
      image: null,
      imagePreview: null,
      errorMessage: null,
      isLoading: false,
      diagnosis: null,
    };
  },
  methods: {
    validateFile(image) {
      if(image.type.split('/')[0] === 'image') {
        return true;
      }
      return false;
    },
    submit() {
      this.diagnosis = null;
      if (this.image) {
        if(this.validateFile(this.image)) {
          this.errorMessage = null;
          this.isLoading = true;

          let reader  = new FileReader();
          reader.addEventListener("load", function () {
            this.imagePreview = reader.result;
          }.bind(this), false);
          reader.readAsDataURL(this.image);

          const formData = new FormData();
          formData.append('image', this.image);
          axios.post('http://localhost:5000/', formData).then((res) => {
            console.log(res.data);
            this.diagnosis = res.data;
            this.isLoading = false;
          })
        } else {
          this.errorMessage = 'Формат файла должен быть JPG/JPEG/PNG';
          this.image = null;
        }
      } else {
        this.errorMessage = 'Загрузите снимок';
      }
    },
  }
};
</script>

<style lang="scss" scoped>

</style>
