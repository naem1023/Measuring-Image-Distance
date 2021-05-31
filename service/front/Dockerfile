FROM node:12.8.1 as builder

ARG REACT_APP_API_URI

ADD ./ /front
WORKDIR /front
RUN npm install
RUN npm run build

FROM nginx:1.13.9-alpine
RUN rm -rf /etc/nginx/conf.d
COPY ./nginx_conf /etc/nginx

COPY --from=builder /front/build /usr/share/nginx/html

EXPOSE 80
CMD nginx -g 'daemon off;'

