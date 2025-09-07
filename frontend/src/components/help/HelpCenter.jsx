import { useState, useEffect } from 'react';
import { XMarkIcon, QuestionMarkCircleIcon, BookOpenIcon, EnvelopeIcon, ArrowRightIcon, ChevronDownIcon, ChevronUpIcon } from '@heroicons/react/24/outline';
import Tutorial from '../tutorial/Tutorial';

const HelpCenter = ({ onClose }) => {
  const [activeTab, setActiveTab] = useState('getting-started');
  const [showTutorial, setShowTutorial] = useState(false);
  const [expandedItems, setExpandedItems] = useState({});

  const toggleItem = (id) => {
    setExpandedItems(prev => ({
      ...prev,
      [id]: !prev[id]
    }));
  };

  const sections = {
    'getting-started': {
      title: 'Getting Started',
      icon: <BookOpenIcon className="h-5 w-5 mr-2" />,
      items: [
        {
          id: 'what-is-beer-game',
          question: 'What is The Beer Game?',
          answer: 'The Beer Game is a simulation that demonstrates the challenges of supply chain management. Players take on different roles in a supply chain (Retailer, Wholesaler, Distributor, or Factory) and make decisions to manage inventory and meet customer demand.'
        },
        {
          id: 'how-to-play',
          question: 'How do I play?',
          answer: '1. Join or create a game\n2. Choose your role in the supply chain\n3. Each turn, decide how many units to order from your supplier\n4. Try to minimize costs while meeting customer demand\n5. The player with the lowest total cost at the end of the game wins!'
        },
        {
          id: 'game-objective',
          question: 'What is the objective?',
          answer: 'The goal is to minimize your total costs, which include:\n- Inventory holding costs\n- Backorder costs\n- Order costs\n\nThe player with the lowest total cost at the end of the game wins.'
        }
      ]
    },
    'gameplay': {
      title: 'Gameplay',
      icon: <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
      </svg>,
      items: [
        {
          id: 'turns',
          question: 'How do turns work?',
          answer: 'Each turn represents one week in the simulation. During your turn, you will:\n1. Receive and process incoming orders from your customer\n2. Update your inventory\n3. Place new orders to your supplier\n4. Receive and process incoming shipments\n5. Ship orders to your customer'
        },
        {
          id: 'costs',
          question: 'What are the costs?',
          answer: 'There are three types of costs in the game:\n\n1. **Inventory Holding Cost**: Cost for each unit in your inventory at the end of each week\n2. **Backorder Cost**: Penalty for each unit of unmet demand\n3. **Order Cost**: Fixed cost for placing an order\n\nYour goal is to minimize the sum of these costs.'
        },
        {
          id: 'lead-time',
          question: 'What is lead time?',
          answer: 'Lead time is the delay between placing an order and receiving it. In this game, there is a 1-week lead time for orders. This means that orders you place this week will arrive in your inventory in 2 weeks (next week for the order to be processed, and the following week for delivery).'
        }
      ]
    },
    'troubleshooting': {
      title: 'Troubleshooting',
      icon: <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
      </svg>,
      items: [
        {
          id: 'connection-issues',
          question: 'I\'m having connection issues',
          answer: 'If you\'re experiencing connection issues, try the following:\n\n1. Check your internet connection\n2. Refresh the page\n3. Clear your browser cache and cookies\n4. Try using a different browser\n\nIf the problem persists, please contact support.'
        },
        {
          id: 'game-not-loading',
          question: 'The game won\'t load',
          answer: 'If the game is not loading, try these steps:\n\n1. Make sure you\'re using a supported browser (Chrome, Firefox, Safari, or Edge)\n2. Clear your browser cache\n3. Disable any ad blockers or extensions that might interfere\n4. Try opening the game in an incognito/private window\n\nIf the issue continues, please contact support with details about your browser and any error messages you see.'
        },
        {
          id: 'lost-password',
          question: 'I forgot my password',
          answer: 'If you\'ve forgotten your password, you can reset it by:\n\n1. Clicking on "Forgot Password" on the login page\n2. Entering your email address\n3. Following the instructions in the email you receive\n\nIf you don\'t receive the email, please check your spam folder.'
        }
      ]
    },
    'contact': {
      title: 'Contact Support',
      icon: <EnvelopeIcon className="h-5 w-5 mr-2" />,
      items: [
        {
          id: 'email-support',
          question: 'Email Support',
          answer: 'For any questions or issues, please email us at support@beergame.example.com. We typically respond within 24 hours.'
        },
        {
          id: 'faq',
          question: 'Frequently Asked Questions',
          answer: 'Check our [FAQ page](https://beergame.example.com/faq) for answers to common questions.'
        },
        {
          id: 'feedback',
          question: 'Send Feedback',
          answer: 'We\'d love to hear your feedback! Please email us at feedback@beergame.example.com with your suggestions or comments.'
        }
      ]
    }
  };

  // Close on Escape key
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  if (showTutorial) {
    return <Tutorial onClose={() => setShowTutorial(false)} />;
  }

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto" aria-labelledby="modal-title" role="dialog" aria-modal="true">
      <div className="flex items-end justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
        <div className="fixed inset-0 bg-gray-500 bg-opacity-75 transition-opacity" aria-hidden="true" onClick={onClose}></div>
        
        <span className="hidden sm:inline-block sm:align-middle sm:h-screen" aria-hidden="true">&#8203;</span>
        
        <div className="inline-block align-bottom bg-white rounded-lg px-4 pt-5 pb-4 text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-4xl sm:w-full sm:p-6">
          <div className="absolute top-0 right-0 pt-4 pr-4">
            <button
              type="button"
              className="bg-white rounded-md text-gray-400 hover:text-gray-500 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
              onClick={onClose}
            >
              <span className="sr-only">Close</span>
              <XMarkIcon className="h-6 w-6" aria-hidden="true" />
            </button>
          </div>
          
          <div className="sm:flex sm:items-start">
            <div className="mt-3 text-center sm:mt-0 sm:text-left w-full">
              <h3 className="text-2xl leading-6 font-bold text-gray-900 mb-6" id="modal-title">
                Help Center
              </h3>
              
              <div className="flex flex-col md:flex-row gap-6">
                {/* Sidebar */}
                <div className="w-full md:w-1/3 space-y-2">
                  <button
                    onClick={() => setShowTutorial(true)}
                    className="w-full flex items-center px-4 py-3 border border-transparent text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                  >
                    <PlayIcon className="-ml-1 mr-2 h-5 w-5" />
                    Start Tutorial
                  </button>
                  
                  <div className="border-b border-gray-200">
                    <nav className="-mb-px flex flex-col space-y-1">
                      {Object.entries(sections).map(([key, section]) => (
                        <button
                          key={key}
                          onClick={() => setActiveTab(key)}
                          className={`${activeTab === key
                            ? 'border-indigo-500 text-indigo-600 bg-indigo-50'
                            : 'border-transparent text-gray-500 hover:text-gray-700 hover:bg-gray-50'
                          } group inline-flex items-center px-3 py-3 border-l-4 text-sm font-medium rounded-r-md`}
                        >
                          {section.icon}
                          {section.title}
                        </button>
                      ))}
                    </nav>
                  </div>
                </div>
                
                {/* Content */}
                <div className="flex-1">
                  <h3 className="text-lg font-medium text-gray-900 mb-4 flex items-center">
                    {sections[activeTab].icon}
                    <span className="ml-2">{sections[activeTab].title}</span>
                  </h3>
                  
                  <div className="space-y-4">
                    {sections[activeTab].items.map((item) => (
                      <div key={item.id} className="border border-gray-200 rounded-lg overflow-hidden">
                        <button
                          onClick={() => toggleItem(item.id)}
                          className="w-full px-4 py-3 text-left text-sm font-medium text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 flex justify-between items-center"
                        >
                          <span>{item.question}</span>
                          {expandedItems[item.id] ? (
                            <ChevronUpIcon className="h-5 w-5 text-gray-400" />
                          ) : (
                            <ChevronDownIcon className="h-5 w-5 text-gray-400" />
                          )}
                        </button>
                        
                        {expandedItems[item.id] && (
                          <div className="px-4 pb-4 pt-2 bg-white">
                            <p className="text-sm text-gray-600 whitespace-pre-line">
                              {item.answer}
                            </p>
                            
                            {item.id === 'how-to-play' && (
                              <button
                                onClick={() => setShowTutorial(true)}
                                className="mt-3 inline-flex items-center text-sm font-medium text-indigo-600 hover:text-indigo-500"
                              >
                                Show me how
                                <ArrowRightIcon className="ml-1 h-4 w-4" />
                              </button>
                            )}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                  
                  {activeTab === 'contact' && (
                    <div className="mt-6 p-4 bg-gray-50 rounded-lg">
                      <h4 className="text-sm font-medium text-gray-900 mb-2">Need more help?</h4>
                      <p className="text-sm text-gray-600 mb-4">
                        If you can't find what you're looking for, our support team is here to help.
                      </p>
                      <button
                        onClick={() => window.location = 'mailto:support@beergame.example.com'}
                        className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                      >
                        <EnvelopeIcon className="-ml-1 mr-2 h-5 w-5" />
                        Contact Support
                      </button>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HelpCenter;
