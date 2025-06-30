import { useState, useEffect, useRef } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { 
  ArrowLeft, 
  Camera, 
  CheckCircle, 
  XCircle,
  RotateCcw,
  Clock
} from 'lucide-react';
import WebcamView from '@/components/WebcamView';
import ExampleAnim from '@/components/ExampleAnim';
import FeedbackDisplay from '@/components/FeedbackDisplay';
import QuizTimer from '@/components/QuizTimer';
import { useLearningData } from '@/hooks/useLearningData';
import { SignWord } from '@/types/learning';

const Session = () => {
  const navigate = useNavigate();
  const { categoryId, chapterId, sessionType } = useParams();
  const { getCategoryById, getChapterById, addToReview } = useLearningData();

  const [data, setData] = useState(null);
  const [currentFrame, setCurrentFrame] = useState(0);

  const [currentSignIndex, setCurrentSignIndex] = useState(0);
  const [isRecording, setIsRecording] = useState(false);
  const [feedback, setFeedback] = useState<'correct' | 'incorrect' | null>(null);
  const [progress, setProgress] = useState(0);
  const [timerActive, setTimerActive] = useState(false);
  const [sessionComplete, setSessionComplete] = useState(false);
  const [quizResults, setQuizResults] = useState<{signId: string, correct: boolean, timeSpent: number}[]>([]);
  const [quizStarted, setQuizStarted] = useState(false);

  const [isPlaying, setIsPlaying] = useState(true); // 자동 재생 활성화
  const [animationSpeed, setAnimationSpeed] = useState(5);
  const animationIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const isQuizMode = sessionType === 'quiz';
  const QUIZ_TIME_LIMIT = 15; // 15초 제한

  const category = categoryId ? getCategoryById(categoryId) : null;
  const chapter = categoryId && chapterId ? getChapterById(categoryId, chapterId) : null;
  const currentSign = chapter?.signs[currentSignIndex];

  useEffect(() => {
    loadData();
  }, []);

  useEffect(() => {
    if (chapter) {
      setProgress((currentSignIndex / chapter.signs.length) * 100);
    }
  }, [currentSignIndex, chapter]);

  // 퀴즈 모드에서 새로운 문제가 시작될 때 자동으로 타이머 시작
  useEffect(() => {
    if (isQuizMode && currentSign && !feedback) {
      setQuizStarted(true);
      setTimerActive(true);
      setIsRecording(true);
      
      // 15초 후 자동으로 시간 초과 처리
      const timer = setTimeout(() => {
        if (isRecording && timerActive) {
          handleTimeUp();
        }
      }, QUIZ_TIME_LIMIT * 1000);

      return () => clearTimeout(timer);
    }
  }, [currentSignIndex, isQuizMode, currentSign, feedback]);

  // 애니메이션 재생/정지 처리
  useEffect(() => {
    if (isPlaying && data && data.pose && data.pose.length > 0) {
      animationIntervalRef.current = setInterval(() => {
        setCurrentFrame(prev => {
          const nextFrame = prev < data.pose.length - 1 ? prev + 1 : 0;
          console.log(`[Session] 프레임 업데이트: ${prev} → ${nextFrame}`);
          return nextFrame;
        });
      }, 1000 / animationSpeed);
    } else {
      if (animationIntervalRef.current) {
        clearInterval(animationIntervalRef.current);
        animationIntervalRef.current = null;
      }
    }

    return () => {
      if (animationIntervalRef.current) {
        clearInterval(animationIntervalRef.current);
      }
    };
  }, [isPlaying, animationSpeed, data]);

    const loadData = async () => {
    try {
      // 첫 번째 JSON 파일만 로드
      const response = await fetch('/result/KETI_SL_0000000414_landmarks.json');
      const landmarkData = await response.json();
      setData(landmarkData);
    } catch (error) {
      console.error('데이터 로드 실패:', error);
    }
  };

  const handleStartRecording = () => {
    setIsRecording(true);
    setFeedback(null);
    
    if (isQuizMode) {
      setTimerActive(true);
    }
    
    // 3초 후 랜덤 피드백 (실제로는 ML 모델 결과)
    setTimeout(() => {
      handleRecordingComplete();
    }, 3000);
  };

  const handleRecordingComplete = () => {
    const isCorrect = Math.random() > 0.3;
    setFeedback(isCorrect ? 'correct' : 'incorrect');
    setIsRecording(false);
    setTimerActive(false);

    if (isQuizMode && currentSign) {
      const timeSpent = QUIZ_TIME_LIMIT - (timerActive ? QUIZ_TIME_LIMIT : 0);
      setQuizResults(prev => [...prev, {
        signId: currentSign.id,
        correct: isCorrect,
        timeSpent
      }]);
      
      if (!isCorrect) {
        addToReview(currentSign);
      }
    }

    // 퀴즈 모드에서는 항상 자동으로 다음 문제로 이동
    if (isQuizMode) {
      setTimeout(() => {
        handleNextSign();
      }, 2000);
    } else if (isCorrect) {
      // 학습 모드에서는 정답일 때 자동으로 다음 수어로 이동
      setTimeout(() => {
        handleNextSign();
      }, 2000);
    }
  };

  const handleTimeUp = () => {
    setIsRecording(false);
    setTimerActive(false);
    setFeedback('incorrect');
    
    if (currentSign) {
      setQuizResults(prev => [...prev, {
        signId: currentSign.id,
        correct: false,
        timeSpent: QUIZ_TIME_LIMIT
      }]);
      addToReview(currentSign);
    }

    // 퀴즈 모드에서는 시간 초과 시에도 자동으로 다음 문제로 이동
    setTimeout(() => {
      handleNextSign();
    }, 2000);
  };

  const handleNextSign = () => {
    if (chapter && currentSignIndex < chapter.signs.length - 1) {
      setCurrentSignIndex(currentSignIndex + 1);
      setFeedback(null);
      setTimerActive(false);
      setQuizStarted(false);
    } else {
      setSessionComplete(true);
    }
  };

  const handleRetry = () => {
    setFeedback(null);
    setIsRecording(false);
    setTimerActive(false);
    setQuizStarted(false);
  };

  if (!category || !chapter || !currentSign) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <h2 className="text-xl font-bold text-gray-800 mb-2">세션을 찾을 수 없습니다</h2>
          <Button onClick={() => navigate('/learn')}>돌아가기</Button>
        </div>
      </div>
    );
  }

  if (sessionComplete) {
    const correctAnswers = quizResults.filter(r => r.correct).length;
    const totalQuestions = quizResults.length;
    
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <Card className="max-w-md w-full mx-4">
          <CardHeader className="text-center">
            <CheckCircle className="h-16 w-16 text-green-600 mx-auto mb-4" />
            <CardTitle>
              {isQuizMode ? '퀴즈 완료!' : '학습 완료!'}
            </CardTitle>
          </CardHeader>
          <CardContent className="text-center space-y-4">
            {isQuizMode && (
              <div className="bg-blue-50 p-4 rounded-lg">
                <h3 className="font-semibold mb-2">결과</h3>
                <p className="text-2xl font-bold text-blue-600">
                  {correctAnswers}/{totalQuestions}
                </p>
                <p className="text-sm text-gray-600">
                  정답률: {Math.round((correctAnswers/totalQuestions) * 100)}%
                </p>
              </div>
            )}
            <p className="text-gray-600">
              '{chapter.title}' {isQuizMode ? '퀴즈를' : '학습을'} 완료했습니다!
            </p>
            <div className="flex space-x-3">
              <Button 
                variant="outline" 
                onClick={() => navigate(`/learn/category/${categoryId}`)}
              >
                챕터 목록
              </Button>
              <Button onClick={() => navigate('/home')}>
                홈으로
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow-sm border-b">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <Button
                variant="ghost"
                onClick={() => navigate(`/learn/category/${categoryId}`)}
                className="hover:bg-blue-50"
              >
                <ArrowLeft className="h-4 w-4 mr-2" />
                뒤로
              </Button>
              <div>
                <h1 className="text-xl font-bold text-gray-800">
                  {isQuizMode ? '퀴즈' : '학습'}: {currentSign.word}
                </h1>
                <p className="text-sm text-gray-600">
                  {chapter.title} • {currentSignIndex + 1}/{chapter.signs.length}
                </p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              {isQuizMode && (
                <div className="flex items-center space-x-2">
                  <Clock className="h-4 w-4 text-blue-600" />
                  <span className="text-sm text-gray-600">퀴즈 모드</span>
                </div>
              )}
              <div className="w-32">
                <Progress value={progress} className="h-2" />
              </div>
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        <div className="max-w-6xl mx-auto">
          {/* 퀴즈 타이머 */}
          {isQuizMode && timerActive && (
            <div className="mb-6">
              <QuizTimer 
                duration={QUIZ_TIME_LIMIT}
                onTimeUp={handleTimeUp}
                isActive={timerActive}
              />
            </div>
          )}

          <div className="grid lg:grid-cols-2 gap-8">
            {/* 퀴즈 모드에서는 예시 영상 대신 텍스트만 표시 */}
            {isQuizMode ? (
              <div className="space-y-4">
                <h3 className="text-lg font-semibold text-gray-800">수행할 수어</h3>
                <Card className="bg-gradient-to-br from-blue-50 to-blue-100">
                  <CardContent className="pt-6">
                    <div className="text-center">
                      <div className="text-6xl mb-6">🤟</div>
                      <h2 className="text-3xl font-bold text-gray-800 mb-4">
                        "{currentSign.word}"
                      </h2>
                      <p className="text-gray-600">
                        위 단어를 수어로 표현해보세요
                      </p>
                      {!quizStarted && (
                        <p className="text-sm text-blue-600 mt-2">
                          퀴즈가 자동으로 시작됩니다
                        </p>
                      )}
                    </div>
                  </CardContent>
                </Card>
              </div>
            ) : (
              <div className="space-y-4">
                <h3 className="text-lg font-semibold text-gray-800">수어 예시</h3>
                {/* <ExampleVideo keyword={currentSign.word} autoLoop={true} /> */}
                <ExampleAnim data={data} currentFrame={currentFrame} showCylinders={true} showLeftHand={true} showRightHand={true}/>
              </div>
            )}

            {/* 웹캠 및 컨트롤 */}
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-gray-800">따라하기</h3>
              <WebcamView isRecording={isRecording} />
              
              <div className="flex justify-center space-x-4">
                {!isQuizMode && !isRecording && !feedback && (
                  <Button 
                    onClick={handleStartRecording}
                    className="bg-green-600 hover:bg-green-700"
                  >
                    <Camera className="h-4 w-4 mr-2" />
                    시작하기
                  </Button>
                )}
                
                {isRecording && (
                  <Button disabled className="bg-red-600">
                    <div className="animate-pulse flex items-center">
                      <div className="w-3 h-3 bg-white rounded-full mr-2" />
                      {isQuizMode ? '퀴즈 진행 중...' : '인식 중...'}
                    </div>
                  </Button>
                )}
                
                {/* 학습 모드에서 오답일 때만 다시 시도 버튼 표시 */}
                {feedback && !isQuizMode && feedback === 'incorrect' && (
                  <div className="flex space-x-2">
                    <Button onClick={handleRetry} variant="outline">
                      <RotateCcw className="h-4 w-4 mr-2" />
                      다시 시도
                    </Button>
                  </div>
                )}
                
                {/* 자동 진행 메시지 */}
                {feedback && (
                  <div className="text-center">
                    {isQuizMode ? (
                      <p className="text-sm text-gray-600">
                        {feedback === 'correct' ? '정답입니다!' : '오답입니다.'} 잠시 후 다음 문제로 넘어갑니다...
                      </p>
                    ) : feedback === 'correct' ? (
                      <p className="text-sm text-green-600">
                        정답입니다! 잠시 후 다음 수어로 넘어갑니다...
                      </p>
                    ) : null}
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* 피드백 */}
          {feedback && (
            <div className="mt-8">
              <FeedbackDisplay feedback={feedback} />
            </div>
          )}
        </div>
      </main>
    </div>
  );
};

export default Session;