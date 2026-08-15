
const res = await fetch('http://127.0.0.1:5173/uploads/0df4772c-fd3/extracted/zidane.jpg');
console.log('Status:', res.status);
console.log('Headers:', res.headers.get('content-type'));
const buf = await res.arrayBuffer();
console.log('Size:', buf.byteLength);

