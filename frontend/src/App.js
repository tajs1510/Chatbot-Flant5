import React, { useState } from 'react';
import { Container, TextField, Button, Typography, Paper, Box } from '@mui/material';
import { motion } from 'framer-motion';
import axios from 'axios';

const ChatBoxAI = () => {
  const [question, setQuestion] = useState('');
  const [file, setFile] = useState(null);
  const [chatLog, setChatLog] = useState([]);

  const handleSend = async (type) => {
    if (!question.trim() && type === 'text') return;

    let response;

    if (type === 'text') {
      response = await axios.post('http://127.0.0.1:5000/generate', { text: question });
    } else if (type === 'file' && file) {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('text', question);
      response = await axios.post('http://127.0.0.1:5000/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
    } else {
      alert('Please provide a valid input');
      return;
    }

    // Cập nhật khung chat
    setChatLog((prev) => [
      ...prev,
      { sender: 'user', message: question },
      { sender: 'ai', message: response.data.answer },
    ]);

    // Reset input
    setQuestion('');
    setFile(null);
  };

  return (
    <Container maxWidth="md" sx={{ mt: 5 }}>
      <Typography variant="h4" gutterBottom align="center">🧠 AI Chat Assistant</Typography>

      <Paper elevation={3} sx={{ height: '60vh', overflow: 'auto', p: 2, mb: 3 }}>
        {chatLog.map((chat, index) => (
          <Box
            key={index}
            sx={{
              textAlign: chat.sender === 'user' ? 'right' : 'left',
              mb: 2,
            }}
          >
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
            >
              <Typography
                variant="body1"
                sx={{
                  display: 'inline-block',
                  p: 1.5,
                  borderRadius: 2,
                  bgcolor: chat.sender === 'user' ? 'primary.main' : 'grey.300',
                  color: chat.sender === 'user' ? 'white' : 'black',
                  maxWidth: '70%',
                  wordWrap: 'break-word',
                }}
              >
                {chat.message}
              </Typography>
            </motion.div>
          </Box>
        ))}
      </Paper>

      <TextField
        label="Enter your question"
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        fullWidth
        sx={{ mb: 2 }}
      />

      <input type="file" accept=".txt" onChange={(e) => setFile(e.target.files[0])} />

      <Box sx={{ display: 'flex', gap: 2, mt: 2 }}>
        <Button variant="contained" color="primary" fullWidth onClick={() => handleSend('text')}>
          Send Question
        </Button>
        <Button variant="contained" color="secondary" fullWidth onClick={() => handleSend('file')}>
          Upload File + Ask
        </Button>
      </Box>
    </Container>
  );
};

export default ChatBoxAI;
